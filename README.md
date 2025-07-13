# Full-Stack Stock Analysis & Prediction Application

This is a full-featured, full-stack web application designed for stock data analysis, historical trend visualization, and machine learning-based predictions. The frontend is built with React for an interactive user experience, while the backend is powered by Python (Flask/FastAPI) to serve data and handle model inference.


## 🚀 Features

*   **Backend Features:**
    *   Scheduled scraping of the latest stock prices and relevant news.
    *   Provides RESTful APIs for the frontend.
    *   Integrates machine learning models to generate stock predictions.
    *   Uses SQLite as a lightweight database for data storage.
*   **Frontend Features:**
    *   Dynamically displays stock K-line charts using a charting library (e.g., Chart.js, ECharts).
    *   Real-time display of stock lists and detailed information.
    *   Responsive design that adapts to various screen sizes.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask (or FastAPI), Pandas, SQLAlchemy, Scikit-learn
*   **Frontend**: React, JavaScript, CSS, Axios
*   **Database**: SQLite
*   **Core Technologies**: Node.js, Python

## 📋 Prerequisites

Before you begin, ensure you have the following software installed on your machine. These steps are for setting up on a clean computer.

1.  **Git**: For cloning the repository.
    *   [Download & Install Git]
2.  **Node.js and npm**: Required to run the frontend application. The LTS (Long-Term Support) version is recommended.
    *   [Download & Install Node.js] (npm is included in the installation)
3.  **Python**: Required to run the backend server. Version 3.8 or higher is recommended.
    *   [Download & Install Python] (During installation, be sure to check the box that says "Add Python to PATH")

## ⚙️ Installation & Setup

Please follow these steps carefully to get the project up and running.

### 1. Clone the Repository

First, open your terminal or command prompt and clone this repository to your local machine.

```bash
git clone https://github.com/your-username/your-repository-name.git
cd stockcode

# Navigate to the backend directory
cd backend

# Create and activate a Python virtual environment (recommended)
# On Windows:
python -m venv venv
.\venv\Scripts\activate

# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Install all required Python packages
pip install -r requirements.txt

# (IMPORTANT) Configure Environment Variables
# Create a file named .env in the backend/ directory.
# You may need to copy from an .env.example if one exists.
# Add any necessary configuration, such as API keys.
# Example .env content:
# API_KEY=YOUR_SECRET_API_KEY
# DATABASE_URL=sqlite:///stock_data.db

# Run the database initialization script (if applicable)
python create_db.py

# Start the backend server
python app.py

# From the root directory (stockcode/), navigate to the frontend directory
cd frontend

# Install all Node.js dependencies
# This might take a few minutes
npm install

# Start the frontend development server
npm start

## 🎉 All Set!
Your development environment is now fully configured. The frontend application will communicate with the backend API to fetch and display data.

## 📝 Project Structure for Your View

stockcode/
├── backend/           
│   ├── .env            # Environment variables 
│   ├── app.py          # Main application entry point (Flask/FastAPI)
│   ├── requirements.txt# Python dependencies list
│   └── ...             # Other business logic, scrapers, etc.
│
└── frontend/           # All frontend code
    ├── node_modules/   # Node.js dependencies (auto-generated)
    ├── public/         # Static assets like index.html and icons
    ├── src/            # React source code
    │   ├── components/ # Reusable React components
    │   ├── App.js      # Main App component
    │   └── ...
    ├── package.json    # Frontend dependencies and script configuration
    └── ...

