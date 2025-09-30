"""
Test script for the FastAPI inference server
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict():
    payload = {"text": "I absolutely love this transformer stack!", "return_all_scores": False}

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nPredict: {response.status_code}")
    result = response.json()
    print(f"Text: {payload['text']}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Inference time: {result['inference_time_ms']:.2f}ms")

    assert response.status_code == 200
    assert "label" in result
    assert "score" in result


def test_predict_with_all_scores():
    payload = {"text": "This is terrible and I hate it.", "return_all_scores": True}

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nPredict with all scores: {response.status_code}")
    result = response.json()
    print(f"Text: {payload['text']}")
    print(f"Primary label: {result['label']} ({result['score']:.4f})")
    print("All scores:")
    for item in result["all_scores"]:
        print(f"  {item['label']}: {item['score']:.4f}")

    assert response.status_code == 200
    assert result["all_scores"] is not None
    assert len(result["all_scores"]) > 0


def test_metrics():
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"\nMetrics: {response.status_code}")
    print("Metrics available ✓")
    assert response.status_code == 200


if __name__ == "__main__":
    print("Testing FastAPI Inference Server")
    print("=" * 50)

    try:
        test_health()
        test_predict()
        test_predict_with_all_scores()
        test_metrics()

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python serving/app.py")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
