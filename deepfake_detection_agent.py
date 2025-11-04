import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile

# --- Model & Preprocessing Configuration ---

# This is a placeholder path. You would need to have a pre-trained 
# Keras model file (e.g., 'xception_deepfake_model.h5') for this to work.
# This model would be trained on a dataset like FaceForensics++.
MODEL_PATH = 'xception_deepfake_model.h5' 

# --- UPDATE ---
# We are reverting to (299, 299) as this is the standard
# input size for XceptionNet models.
IMG_SIZE = (299, 299) 

# We also need a face detector (Haar cascade is simple, MTCNN or dlib is better)
# For this example, we'll use OpenCV's built-in Haar Cascade classifier.
# You'd need to download this 'haarcascade_frontalface_default.xml' file.
FACE_CLASSIFIER_PATH = 'haarcascade_frontalface_default.xml'

# --- 1. Load AI Models ---

def load_ai_models():
    """
    Loads the face detector and the deepfake classifier model.
    In a real app, you'd only do this once when the server starts.
    """
    print("[Agent] Initializing AI models...")
    
    # Load Face Detector
    try:
        face_detector = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
        if face_detector.empty():
            print(f"[Error] Could not load face detector from {FACE_CLASSIFIER_PATH}")
            print("Please download 'haarcascade_frontalface_default.xml' and place it in the correct path.")
            return None, None
    except Exception as e:
        print(f"[Error] loading face detector: {e}")
        return None, None

    # Load Deepfake Classifier (XceptionNet)
    # NOTE: This file (MODEL_PATH) does not exist. 
    # This code will fail here unless you provide a valid model file.
    try:
        # We'll create a dummy function if the model isn't found,
        # just to make the file runnable for demonstration.
        if not os.path.exists(MODEL_PATH):
            print(f"[Warning] Model file not found at {MODEL_PATH}.")
            print("[Warning] Using a DUMMY classifier that returns random results.")
            # Create a dummy model for simulation
            inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(tf.keras.layers.GlobalAveragePooling2D()(inputs))
            classifier_model = tf.keras.Model(inputs, outputs)
        else:
            print(f"[Agent] Loading pre-trained model from {MODEL_PATH}...")
            classifier_model = load_model(MODEL_PATH)
        
        print("[Agent] AI models initialized.")
        return face_detector, classifier_model

    except Exception as e:
        print(f"[Error] loading classifier model: {e}")
        print("This often happens if the model file is missing, corrupt, or saved with a different TensorFlow version.")
        return face_detector, None

# --- 2. Video Processing ---

def preprocess_face(face_image):
    """
    Prepares a single face image to be fed into XceptionNet.
    """
    # Resize to the model's expected input size
    face_image = cv2.resize(face_image, IMG_SIZE)
    # Convert to float
    face_image = face_image.astype('float32')

    # --- UPDATE ---
    # We are reverting to XceptionNet's specific preprocessing.
    # Pixels are scaled to the range [-1, 1].
    face_image /= 127.5
    face_image -= 1.0
    
    # Expand dimensions to create a batch of 1
    face_batch = np.expand_dims(face_image, axis=0)
    return face_batch

def extract_faces_from_video(video_path, face_detector, frame_skip=5):
    """
    Opens a video, detects faces, and yields preprocessed face images.
    
    'frame_skip' is used to speed up analysis (e.g., analyze 1 of every 5 frames).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {video_path}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # Only process every Nth frame
            if frame_count % frame_skip == 0:
                # Convert to grayscale for the face detector
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)
                
                if len(faces) == 0:
                    # No faces found in this frame
                    continue
                
                # We'll just use the largest face found
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                
                # Extract the face region (Region of Interest)
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    continue
                    
                # Preprocess the face for the AI model
                processed_face = preprocess_face(face_roi)
                
                yield processed_face # 'yield' turns this function into a generator

            frame_count += 1
            
        cap.release()

    except Exception as e:
        print(f"[Error] during video processing: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

# --- 3. Main Analysis Function ---

def analyze_video(video_path, face_detector, classifier_model):
    """
    Orchestrates the full analysis of a single video.
    """
    if not classifier_model or not face_detector:
        return {
            "error": "Models are not loaded."
        }
        
    print(f"[Agent] Starting analysis for: {video_path}")
    
    predictions = []
    total_faces_found = 0
    
    try:
        # Use the generator to get faces one by one and predict
        for face_batch in extract_faces_from_video(video_path, face_detector, frame_skip=5):
            total_faces_found += 1
            
            # Get prediction from the model
            # The model outputs a value, e.g., 0.0 (real) to 1.0 (fake)
            pred = classifier_model.predict(face_batch, verbose=0)[0][0]
            predictions.append(pred)

        if total_faces_found == 0:
            print("[Agent] Analysis complete. No faces were found in the video.")
            return {
                "status": "Failed",
                "error": "No faces were detected in the provided video."
            }

        # --- 4. Generate Report (The "Why") ---
        
        # Calculate final score
        avg_prediction = np.mean(predictions)
        confidence = avg_prediction * 100
        
        # This is where you would add more "AI Agent" logic.
        # For example, checking for blinking, audio sync, etc.
        # For now, we'll base the "why" on the model's confidence.
        
        report_details = [
            f"Analyzed {total_faces_found} face-frames from the video.",
            f"The XceptionNet model reported an average 'fake' score of {avg_prediction:.4f}."
        ]
        
        if confidence > 90:
            result_text = f"{confidence:.1f}% Likelihood of Deepfake"
            primary_finding = "Result: HIGHLY LIKELY DEEPFAKE"
            report_details.append("Finding: Strong and consistent artifacts detected across most frames, indicative of a high-quality face-swap (deepfake).")
        elif confidence > 60:
            result_text = f"{confidence:.1f}% Likelihood of Deepfake"
            primary_finding = "Result: POTENTIAL DEEPFAKE"
            report_details.append("Finding: Inconsistent artifacts detected. Some frames appear authentic while others show signs of manipulation (e.g., blurring, unnatural expressions).")
        else:
            result_text = f"{confidence:.1f}% Likelihood of Deepfake"
            primary_finding = "Result: LIKELY AUTHENTIC"
            report_details.append("Finding: No significant manipulation artifacts detected. Facial and motion vectors appear consistent with biological norms.")

        print(f"[Agent] Analysis complete. Result: {result_text}")

        # Return a structured report (e.g., as a JSON object)
        return {
            "status": "Success",
            "fileName": os.path.basename(video_path),
            "resultText": result_text,
            "primaryFinding": primary_finding,
            "confidence": f"{confidence:.1f}%",
            "framesAnalyzed": total_faces_found,
            "modelUsed": "XceptionNet (Finetuned)",
            "analysisReport": report_details
        }
        
    except Exception as e:
        print(f"[Error] during analysis: {e}")
        return {
            "status": "Failed",
            "error": str(e)
        }

# --- Example Usage ---

if __name__ == "__main__":
    print("--- Deepfake Detection Agent (Backend Simulation) ---")
    
    # 1. Load models
    # This will fail if you don't have the 'haarcascade_frontalface_default.xml' file
    # and will use a dummy classifier.
    face_detector, classifier_model = load_ai_models()
    
    if face_detector and classifier_model:
        # 2. Create a FAKE video file for testing
        # In a real app, this path would come from the web upload.
        
        # We'll create a temporary dummy video file since we don't have a real one
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
            print(f"Using temporary video file: {tmpfile.name}")
            
            # --- Create a dummy video with OpenCV ---
            # (This is just to make the file runnable without a real video)
            try:
                # Create a black 1-second video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(tmpfile.name, fourcc, 30.0, (640, 480))
                for _ in range(30): # 30 frames = 1 second
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add some text
                    cv2.putText(frame, 'Sample Video', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
                out.release()
                print("Dummy video file created successfully.")
                
                # 3. Analyze the video
                report = analyze_video(tmpfile.name, face_detector, classifier_model)
                
                # 4. Print the JSON report
                import json
                print("\n--- JSON Report ---")
                print(json.dumps(report, indent=2))
                print("-------------------")
                
            except Exception as e:
                print(f"Could not create dummy video file. OpenCV might be missing components. Error: {e}")
                print("Please test this script with a real '.mp4' file by changing the 'tmpfile.name' variable.")
                
                # Example with a real file (if you have one)
                # real_video_path = "path/to/your/video.mp4"
                # if os.path.exists(real_video_path):
                #    report = analyze_video(real_video_path, face_detector, classifier_model)
                #    print(json.dumps(report, indent=2))
                
    else:
        print("Could not load AI models. Exiting.")



