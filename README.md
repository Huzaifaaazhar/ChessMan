# Chess Piece Recognition Application

## Problem Statement

In the realm of chess, quick and accurate recognition of chess pieces can enhance gameplay, coaching, and automated analysis. This project addresses the need for an intelligent system capable of recognizing chess pieces from images, which can be particularly useful in chess training tools, game analysis, and educational applications.

The goal is to develop a web application that uses machine learning to classify images of chess pieces. The application provides a user-friendly interface for uploading images and receiving predictions about the type of chess piece in the image. This system leverages TensorFlow for model training and FastAPI for serving the application.

## Features

- **Image Classification**: Utilizes a deep learning model to classify images of chess pieces into one of six categories: Bishop, King, Knight, Pawn, Queen, or Rook.
- **User Interface**: Simple and intuitive web interface for uploading images and viewing predictions.
- **Model Training**: Includes code for training the classification model using ResNet50 and saving the best-performing model.
- **Deployment**: Instructions for deploying the application on Azure Static Web Apps.

## Technologies

- **Backend**: FastAPI
- **Machine Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS
- **Deployment**: Azure Static Web Apps

## Directory Structure

```
Chessman/
│
├── Chess/
│   ├── Bishop/
│   ├── King/
│   ├── Knight/
│   ├── Pawn/
│   ├── Queen/
│   └── Rook/
│
├── env/                      # Virtual environment
├── templates/
│   └── index.html            # Frontend HTML file
├── app.py                    # FastAPI application code
├── best_chess_model.keras    # Trained model file
├── final_chess_model.h5      # Final model file
├── model.ipynb               # Jupyter notebook for model training
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

## Setup

### Prerequisites

- Python 3.12 or newer
- Virtual Environment (optional but recommended)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/chessman.git
   cd chessman
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download or Train the Model**

   If you have not yet trained the model, you can do so by running `model.ipynb` in a Jupyter environment. Ensure that `best_chess_model.keras` and `final_chess_model.h5` are present in the root directory.

### Running the Application

1. **Start the FastAPI Server**

   ```bash
   uvicorn app:app --reload
   ```

2. **Open Your Browser**

   Navigate to `http://127.0.0.1:8000` to access the web interface.

## Usage

1. **Upload an Image**

   On the homepage, use the form to upload an image of a chess piece.

2. **View Prediction**

   After submitting the image, the application will process it and display the predicted chess piece category.

## Deployment on Azure Static Web Apps

1. **Create a GitHub Repository**

   Push your project to GitHub if it is not already there.

2. **Create a Static Web App on Azure**

   Follow the [Azure documentation](https://learn.microsoft.com/en-us/azure/static-web-apps/getting-started?tabs=visual-studio-code) to create a Static Web App and connect it to your GitHub repository.

3. **Configure Azure Deployment**

   Azure Static Web Apps uses a `staticwebapp.config.json` file for configuration. Add this file to your project with the necessary configurations for your FastAPI backend and frontend.

4. **Deploy**

   Commit and push your changes to GitHub. Azure Static Web Apps will automatically build and deploy your application.

## Contributing

Feel free to fork the repository and submit pull requests. If you find any issues or have suggestions for improvements, please open an issue on GitHub.

## Acknowledgements

- TensorFlow and Keras for the machine learning framework.
- FastAPI for the backend framework.
- [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) for the pre-trained model.
