# CNN Project: Cat vs Dog Image Classification

Welcome to the Cat vs Dog Image Classification project! This repository contains a project demonstrating the use of Convolutional Neural Networks (CNNs) to classify images as either a cat or a dog.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

This project focuses on building a CNN model to classify images into two categories: cats and dogs. We use TensorFlow and Keras to train a deep neural network on a dataset of labeled images.

- **Technologies Used**: TensorFlow, Keras
- **Dataset**: [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Key Features**: Data preprocessing, model training, evaluation, and inference.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NicolasIDias/Convolutional-neural-network.git
    cd Convolutional-neural-network
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and extract it into a directory named `data`.

2. Navigate to the project directory:

    ```bash
    cd cat-vs-dog-classification
    ```

3. Run the training script:

    ```bash
    python train.py
    ```

4. To evaluate the model, run:

    ```bash
    python evaluate.py
    ```

5. To classify a new image, run:

    ```bash
    python classify.py --image-path path_to_your_image
    ```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or pull requests, please feel free to submit them.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
