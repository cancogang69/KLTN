import requests
import base64
from io import BytesIO
from PIL import Image
import json

def debug_gradio_api(
    gradio_url="https://2a5e580974905d5019.gradio.live",
    input_image_path=None
):
    """
    Comprehensive debugging function for Gradio API
    """
    # Prepare headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': gradio_url,
        'Referer': gradio_url
    }

    # Load and convert image if provided
    if input_image_path:
        with open(input_image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        encoded_image = None

    # Attempt different payload structures
    payload_variations = [
        # Variation 1: Simple payload
        {
            "data": [
                encoded_image,  # init image
                "High quality artistic rendering",  # prompt
                "low quality, blurry",  # negative prompt
                0.5,  # denoising strength
                7,  # cfg scale
                20  # steps
            ]
        },
        # Variation 2: More detailed payload
        {
            "fn_index": 1,  # if specific function index is required
            "data": [
                encoded_image,
                "High quality artistic rendering",
                "low quality, blurry",
                0.5, 7, 20, 512, 512
            ]
        }
    ]

    # Try different API endpoints
    endpoints = [
        f"{gradio_url}api/predict",
        f"{gradio_url}run/predict",
        f"{gradio_url}api/queue/join"
    ]

    for endpoint in endpoints:
        for payload in payload_variations:
            try:
                # print(f"Attempting endpoint: {endpoint}")
                # print(f"Payload: {json.dumps(payload, indent=2)}")

                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                print(f"Response Status Code: {response.status_code}")
                print("Response Headers:")
                for key, value in response.headers.items():
                    print(f"{key}: {value}")

                # Try to print response content for debugging
                try:
                    response_json = response.json()
                    print("Response JSON:")
                    print(json.dumps(response_json, indent=2))
                except Exception as json_error:
                    print(f"Could not parse JSON: {json_error}")
                    print("Response Text:")
                    print(response.text)

                response.raise_for_status()

            except requests.RequestException as e:
                print(f"Request to {endpoint} failed: {e}")
                continue

def main():
    # Replace with your actual image path
    input_image_path = "C:\\Users\\ductu\\Downloads\\mwq17268.png"

    debug_gradio_api(
        gradio_url="https://bb1a01295cadd73890.gradio.live/",
        input_image_path=input_image_path
    )

if __name__ == "__main__":
    main()