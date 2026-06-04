"""Test Ollama from Python (same way your app does)"""
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def test_ollama(model, prompt, timeout=300):
    print(f"🔄 Testing {model}...")
    start = time.time()
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0,
                "num_predict": 30
            },
            timeout=timeout
        )
        elapsed = time.time() - start
        result = response.json().get("response", "").strip()
        
        print(f"✅ {model} responded in {elapsed:.1f}s")
        print(f"📝 Output: {result[:150]}...")
        return True
        
    except requests.exceptions.Timeout:
        print(f"❌ {model} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {model} error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Ollama from Python (like your app)...\n")
    
    # Test your exact workflow
    test_ollama("phi3:mini", "Classify: Ask a database question. Return GENERAL or DATABASE.")
    print()
    test_ollama("mistral:7b", "Clarify: Show records matching the user's request. Keep meaning.")
    print()
    print("⚠️  Next test may take 30-90s (first load of deepseek-coder):")
    test_ollama("deepseek-coder:6.7b", "Generate SQL for a generic database question. ONLY SQL, no markdown.")
