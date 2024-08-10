# NLP Project

This repository contains a FastAPI application that provides two main endpoints for performing Named Entity Recognition (NER) and Part-of-Speech (POS) tagging using a pre-trained ONNX model.

## API Endpoints

### **/infer (POST)**

**Description:**  
This endpoint performs inference on the provided text using the saved ONNX model.

- **Request Body:**
  - `text` (string): The input text to be processed.

- **Response:**
  - A JSON response containing the tokens along with their predicted POS and NER tags.

### **/start_training (GET)**

**Description:**  
This endpoint initiates the training and evaluation pipeline. It initiates training process, then saves the model and finally evaluate the model on test set based on several performance metrices.


## How to Run

Follow these steps to set up the environment and run the application:

### 1. Clone the GitHub Repository

```bash
git clone https://github.com/ShakilMahmudShuvo/GTL-MLE-Assignment-Submission.git
```
### 2. Create a Virtual Environment

To create a virtual environment, run the following command:

```bash
python -m venv venv
```
### 3. Activate the Virtual Environment

- **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```
### 4. Installing pre-requisite libraries:
  - In the base folder run 
```
pip install -r requirements.txt
```
## Running API
* To run the API run in the base folder's terminal:
```
python app.py
```

