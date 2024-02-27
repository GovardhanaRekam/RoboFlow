# Tomato Disease Leaf Prediction using Roboflow and OpenCV

This project demonstrates real-time prediction of tomato leaf diseases using a pre-trained model from Roboflow and OpenCV with Python. It captures images of tomato leaves from a webcam, sends them to Roboflow for prediction, and overlays bounding boxes on detected diseased areas before displaying them.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- Roboflow Python Library (`roboflow`)

## Installation

1. Clone the repository


3. Obtain a Roboflow API key and replace `"YOUR_API_KEY"` in the code with your actual API key.

## Usage

1. Run the script:

    ```bash
    python tomato_disease_prediction.py
    ```

2. Press 'q' to quit the application.

## Customization

- Adjust the confidence threshold and overlap parameters in the code (`confidence`, `overlap`) according to your needs.
- Modify the Roboflow project and model versions as required.

## Contributing

Contributions are welcome! Fork the repository and submit a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).
