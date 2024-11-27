# Live Retina Authentication System

This project implements a simple biometric authentication system that uses the **eye's features (iris)** to authenticate users. The system takes a picture of the user's eye, extracts key features, stores them, and then compares those features for subsequent access attempts.

## Features
- **Capture Eye Image:** The system takes a picture of the user's eye using the webcam and stores it as a reference image.
- **Feature Extraction:** It extracts unique features from the captured eye using the **ORB (Oriented FAST and Rotated BRIEF)** feature detector.
- **Authentication:** For future access attempts, it captures a live image of the user's eye, compares the features to the stored reference, and authenticates the user.
- **Real-time Authentication:** The system continuously checks the webcam feed and compares features as the user attempts to authenticate by pressing a key.

