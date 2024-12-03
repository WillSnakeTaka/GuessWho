import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from gtts import gTTS
import os
import random

# Load your trained model
model = load_model("gesture_recognition_model.h5")

# Define labels for your gestures
gesture_classes = ["Mice", "Cat", "The Fool"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Streamlit app layout
st.title("Gesture Recognition and Mini-Game")
st.image("Kitten.jpeg", 
         caption="Game Rules: Cat disarms Mice, Mice disturb The Fool, The Fool pets the Cat", 
         use_container_width=True)
st.write("Let's guess (Mice, Cat, The Fool) in front of your webcam and play the game!")

# Function for Text-to-Speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")
    os.system("afplay temp.mp3")  # macOS
    

# Function to preprocess landmarks for the model (Those hand dots)
def preprocess_landmarks(landmarks):
    # Convert landmarks to a single flattened array
    flattened = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    # Add batch dimension
    return np.expand_dims(flattened, axis=0)

# Random gesture generator
def generate_random_gesture():
    return random.choice(gesture_classes)

# Determine game result
def resolve_game(player_gesture, computer_gesture):
    if player_gesture == computer_gesture:
        return "It's a tie!"
    elif (player_gesture == "Cat" and computer_gesture == "Mice") or \
         (player_gesture == "Mice" and computer_gesture == "The Fool") or \
         (player_gesture == "The Fool" and computer_gesture == "Cat"):
        return "You win!"
    else:
        return "Computer wins!"

# Video capture function
def run_gesture_recognition():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for displaying video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocess landmarks for gesture recognition
                landmarks_array = preprocess_landmarks(hand_landmarks)

                # Predict gesture
                predictions = model.predict(landmarks_array)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)
                predicted_label = gesture_classes[predicted_class]

                # Draw a rectangle background
                cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)  # Black rectangle

                # Display prediction
                cv2.putText(frame, f"{predicted_label} ({confidence:.2f})",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

                # Speak gesture
                if confidence > 0.5:
                    speak(f"You choose a {predicted_label}")

                # Mini-game
                computer_gesture = generate_random_gesture()
                game_result = resolve_game(predicted_label, computer_gesture)

                # Display computer gesture and game result
                cv2.putText(frame, f"Computer: {computer_gesture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Result: {game_result}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Announce game result
                speak(game_result)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR")

        # Stop the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Run the gesture recognition function
if st.button("Start The Magic 🐍"):
    run_gesture_recognition()
