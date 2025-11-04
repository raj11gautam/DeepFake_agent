import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import werkzeug.utils

# Import the logic from your AI agent file
import deepfake_detection_agent as agent

# --- 1. Initialize Flask App & AI Models ---

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (allows browser to talk to server)

print("---* Server is starting *---")
print("Loading AI models... This may take a moment.")
# Load the models once when the server starts
try:
    face_detector_model, classifier_model = agent.load_ai_models()
    if not face_detector_model or not classifier_model:
        print("\n[FATAL ERROR] Could not load AI models. The server will not work correctly.")
        print("Please ensure 'haarcascade_frontalface_default.xml' is present.")
        print("If you have a real model, ensure 'xception_deepfake_model.h5' is present.")
    else:
        print("\n[SUCCESS] AI models loaded. Server is ready at http://127.0.0.1:5000")
except Exception as e:
    print(f"\n[FATAL ERROR] An exception occurred during model loading: {e}")
    face_detector_model = None
    classifier_model = None
print("---* Server startup complete *---")


# --- 2. Define the API Endpoint ---

@app.route('/analyze', methods=['POST'])
def handle_analysis_request():
    """
    This function is called when the frontend sends a video to the /analyze URL.
    """
    print("\n[Server] Received new analysis request.")
    
    # Check if models are loaded
    if not face_detector_model or not classifier_model:
        print("[Server] Error: Models are not loaded. Sending error response.")
        return jsonify({"status": "Failed", "error": "Server-side models are not loaded. Check server logs."}), 500

    # 1. Check if the 'video' file is in the request
    if 'video' not in request.files:
        print("[Server] Error: 'video' part missing in request. Sending error response.")
        return jsonify({"status": "Failed", "error": "No video file provided in the request."}), 400

    file = request.files['video']

    # 2. Check if the file has a name
    if file.filename == '':
        print("[Server] Error: No file selected. Sending error response.")
        return jsonify({"status": "Failed", "error": "No file selected."}), 400

    if file:
        # 3. Save the file temporarily
        filename = werkzeug.utils.secure_filename(file.filename)
        # We use a temporary directory to save the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, filename)
            try:
                file.save(video_path)
                print(f"[Server] Video saved temporarily to: {video_path}")

                # 4. Run the analysis using the imported agent
                print(f"[Server] Calling AI agent to analyze video...")
                report = agent.analyze_video(video_path, face_detector_model, classifier_model)
                print(f"[Server] Analysis complete. Sending report to frontend.")

                # 5. Return the JSON report to the frontend
                return jsonify(report)

            except Exception as e:
                print(f"[Server] An internal error occurred during analysis: {e}")
                return jsonify({"status": "Failed", "error": f"An internal server error occurred: {e}"}), 500
            
            # The temporary directory and its contents are automatically deleted here
    
    return jsonify({"status": "Failed", "error": "Unknown error occurred."}), 500

# --- 3. Run the Server ---

if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    # 'debug=True' makes the server auto-reload when you save changes
    app.run(debug=True, port=5000)
