Deepfake Detection System

A complete web application designed to analyze video files and detect potential deepfakes using an AI backend. The system features a modern, responsive frontend for video uploads and a powerful Python server running an Xception-based model for real-time analysis.

✨ Features

Modern Web Interface: A sleek, responsive, and theme-able UI built with Tailwind CSS, featuring frosted glass effects.

Video Upload: Drag-and-drop or browse to upload video files (.mp4, .mov, etc.).

Real-Time AI Analysis: Connects to a Python backend to analyze videos using a deep learning model (Xception).

Detailed Reporting: Displays a comprehensive report including:

Confidence Score (Fake vs. Real)

AI Analysis Report (Suspicious findings)

Key Anomaly Statistics

PDF Report Generation: Download a professional, formatted PDF of the analysis report.

User Authentication: (Optional) Firebase/Local simulation for user login, sign-up, and profile management.

Analysis History: (Optional) Users can view their past analysis results (requires Firebase).

🚀 Architecture

The project is built on a client-server architecture:

Frontend (Client): deepfake_detector.html

A single-page application built with plain HTML, Tailwind CSS, and modern JavaScript (ES6+ Modules).

Handles all user interactions, file uploads, and renders the analysis report.

Communicates with the backend via fetch API calls.

Backend (Server): server.py / deepfake_detection_agent.py

A Python server (likely using Flask or FastAPI) that exposes an /analyze endpoint.

Receives the video file, processes it using OpenCV, and feeds it into the Xception model.

Returns a JSON response with the analysis results.

🛠️ Tech Stack

Frontend:

HTML5

Tailwind CSS

JavaScript (ES6+)

jsPDF (for PDF generation)

Firebase (for optional authentication & database)

Backend:

Python 3.x

Flask (or FastAPI)

TensorFlow / Keras

OpenCV (cv2)

Flask-CORS (Required)

⚙️ Setup and Installation

To run this project, you must set up both the backend server and the frontend application.

1. Backend Setup (Python Server)

Clone the Repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name


Create a Python Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

Create a requirements.txt file with the following content (you may need to add/remove libraries based on your code):

flask
flask-cors
tensorflow
opencv-python-headless
numpy
# Add any other libraries your agent needs


Install the dependencies:

pip install -r requirements.txt


Add the AI Model:

--> Download your trained Xception model file.

--> Place it in the same directory as your Python server script.

--> Rename the model file to xception_deepfake_model.h5 (or update the name in your Python script to match).

Run the Server:

Run the Python script that starts your Flask/FastAPI server (e.g., server.py).

python server.py


Wait until you see a message confirming the server is running, like:
[SUCCESS] AI models loaded. Server is ready at http://127.0.0.1:5000

2. Frontend Setup (HTML Web App)

Open the HTML File:

--> You can run the deepfake_detector.html file in two ways:

  - Recommended (VS Code): Install the "Live Server" extension, right-click on deepfake_detector.html, and select "Open with Live Server".

  - Simple: Simply double-click the deepfake_detector.html file to open it in your browser.

Verify Backend URL (CRITICAL):

--> Open deepfake_detector.html in your code editor.

--> Go to approximately line 950 (inside the <script type="module"> tag).

--> Find the BACKEND_URL variable and make sure it matches your running Python server address:

// Make sure this URL matches your running Python server
const BACKEND_URL = "[http://127.0.0.1:5000/analyze](http://127.0.0.1:5000/analyze)"; 


usage

1.Start the Python Backend server first (e.g., python server.py).

2.Open the deepfake_detector.html file in your browser.

3.Drag and drop a video file (or use "Browse Files").

4.Click the "Analyze Video" button.

5.Wait for the analysis to complete.

6.View the detailed report and download the PDF.

🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find a bug or have a feature request.

1.Fork the Project

2.Create your Feature Branch (git checkout -b feature/AmazingFeature)

3.Commit your Changes (git commit -m 'Add some AmazingFeature')

4.Push to the Branch (git push origin feature/AmazingFeature)

5.Open a Pull Request

📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details (you will need to create this file).
