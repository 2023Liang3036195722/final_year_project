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

2.  **Node.js and npm**: Required to run the frontend application. The LTS (Long-Term Support) version is recommended.(npm is included in the installation)
3.  **Python**: Required to run the backend server. Version 3.8 or higher is recommended. (During installation, be sure to check the box that says "Add Python to PATH")

## ⚙️ Installation & Setup

Please follow these steps carefully to get the project up and running.

### 1. Clone the Repository

First, open your terminal or command prompt and clone this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd stockcode
   ```

### 2. Backend Setup & Launch

First, set up and run the backend server.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment:**
    *   On **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On **macOS / Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install all required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(IMPORTANT) Configure Environment Variables:**
    Create a file named `.env` in the `backend/` directory. You may need to copy from an `.env.example` if one exists. Add any necessary configuration, such as API keys.
    
    *Example `.env` content:*
    ```
    API_KEY="YOUR_SECRET_API_KEY"
    DATABASE_URL="sqlite:///stock_data.db"
    ```

5.  **Start the backend server:**
    ```bash
    python app.py
    ```
    > Keep this terminal window open. The backend server needs to be running for the frontend to work.

### 3. Frontend Setup & Launch

Now, in a **new terminal window**, set up and run the frontend application.

1.  **Navigate to the frontend directory (from the root `stockcode/` folder):**
    ```bash
    cd frontend
    ```

2.  **Install all Node.js dependencies:**
    This might take a few minutes.
    ```bash
    npm install
    ```

3.  **Start the frontend development server:**
    ```bash
    npm start
    ```

## 🎉 All Set!

Your development environment is now fully configured. The frontend application (usually at `http://localhost:3000`) will communicate with the backend API to fetch and display data.



