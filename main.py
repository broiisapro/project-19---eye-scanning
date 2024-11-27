import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# ORB detector
orb = cv2.ORB_create()

def capture_eye_image():
    """Capture a picture of the eye and save it."""
    print("Please look at the camera. Press 'q' to capture your eye image.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the video feed
        cv2.imshow("Capture Eye Image", frame)
        
        # Wait for 'q' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save the image when 'q' is pressed
            eye_image = frame
            cv2.imwrite("reference_eye.jpg", eye_image)
            print("Eye image captured and saved.")
            break
    
    return eye_image

def extract_features(image):
    """Extract ORB features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def store_features(des):
    """Store the extracted features to a file."""
    np.save("stored_features.npy", des)

def load_stored_features():
    """Load the stored features from file."""
    try:
        stored_features = np.load("stored_features.npy")
        return stored_features
    except FileNotFoundError:
        print("No stored features found. Please capture an image first.")
        return None

def match_features(des1, des2):
    """Match the features between two images using the BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors between the two sets
    matches = bf.match(des1, des2)
    
    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def authenticate_eye(eye_image):
    """Authenticate the user by comparing the features of the captured eye to the stored reference."""
    # Extract features from the new eye image
    kp1, des1 = extract_features(eye_image)
    
    # Load stored features
    stored_features = load_stored_features()
    
    if stored_features is None:
        print("No stored features to compare with. Please capture the eye image first.")
        return False
    
    # Match the features
    matches = match_features(des1, stored_features)
    
    # Determine the similarity based on the number of good matches
    good_matches = [m for m in matches if m.distance < 50]  # Distance threshold
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # If the number of good matches is above a certain threshold, consider it authenticated
    if len(good_matches) > 10:
        print("Authentication successful!")
        return True
    else:
        print("Authentication failed!")
        return False

def start_video_stream():
    """Start the webcam stream to capture the eye for authentication."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Display the video feed
        cv2.imshow("Real-time Authentication", frame)
        
        # Wait for the 'c' key to authenticate
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Capture and authenticate the eye
            print("Capturing and authenticating...")
            authenticated = authenticate_eye(frame)
            if authenticated:
                print("Access Granted!")
            else:
                print("Access Denied!")
        
        # Wait for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Capture the eye image and store the features
    capture_eye_image()
    reference_image = cv2.imread("reference_eye.jpg")
    kp, des = extract_features(reference_image)
    store_features(des)
    
    # Start the video stream for real-time authentication
    start_video_stream()
