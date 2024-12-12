# Project eKYC

This project is an implementation of an electronic Know Your Customer (eKYC) system that utilizes face verification and ID card OCR (Optical Character Recognition) technology.

## Features

- Face verification: The system uses fFacenet model to verify the identity of the user by comparing the user's face with the face on their ID card.
- ID card OCR: The system extracts information from ID cards.

## Installation


1. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

2. Run the application:

    ```shell
    python main.py
    ```

2. Follow the on-screen instructions to perform face verification and ID card OCR.

## Generating the dataset

1. To generate the data, run the following command:

    ```shell
    python gendata.py
    ```
2. To generate dataset for training, run the following command:

    ```shell
    python label_gen.py
    ```
