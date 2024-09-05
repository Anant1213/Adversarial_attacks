# Adversarial Attacks on Deep Learning Models

This project provides a web-based application where users can test the robustness of deep learning models against various adversarial attacks like **FGSM (Fast Gradient Sign Method)** and **One-Pixel Attack**. The application is built using **Streamlit** for the frontend and **FastAPI** for the backend, providing a simple interface for uploading images and processing them with adversarial perturbations.

## Features
- **FGSM Attack**: Modify images by adding small perturbations to fool deep learning models.
- **One-Pixel Attack**: Randomly modify individual pixels to cause misclassification in the model.

## Project Structure

Adversarial_Attack_Project/ │ ├── backend/ │ ├── attacks/ │ │ ├── fgsm.py │ │ └── one_pixel_attack.py │ ├── app.py (backend logic with FastAPI) │ └── model.py (loading and processing model) │ ├── frontend/ │ └── app.py (frontend logic using Streamlit) │ └── 


### Backend
The **backend** is built using **FastAPI** and exposes an API for performing adversarial attacks on uploaded images.

### Frontend
The **frontend** is built using **Streamlit**, providing a user-friendly interface for uploading images, selecting attack types, and visualizing results.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Anant1213/adversarial_attack.git
cd adversarial-attack-project
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Command for running backend
   ```bash
   cd backend
   uvicorn app:app --reload
   ```
4. Start the frontend
   ```bash
   cd frontend
   streamlit run app.py
   ```

### Contributing
Feel free to submit issues, fork the repository, and create pull requests to contribute to this project.



