import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the camera
cap = cv2.VideoCapture(0)

# Counter for saving images
image_counter = 0

# Fixed size for the hand window
fixed_width = 300
fixed_height = 300

# Create a window with fixed size (though OpenCV doesn't strictly enforce this)
cv2.namedWindow('Cropped Hand', cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the RGB image.
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # Calculate the bounding box around the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

            # Add some padding around the hand
            padding = 20
            x_min = max(0, int(x_min - padding))
            x_max = min(image.shape[1], int(x_max + padding))
            y_min = max(0, int(y_min - padding))
            y_max = min(image.shape[0], int(y_max + padding))

            # Crop the hand from the original image
            hand_image = image[y_min:y_max, x_min:x_max]

            # Resize the hand image to a fixed size
            resized_hand_image = cv2.resize(hand_image, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)

            # Display the resized hand image
            cv2.imshow('Cropped Hand', resized_hand_image)

    # Display the resulting image
    cv2.imshow('MediaPipe Hand Detection', image)

    # Press 'c' to capture and convert to black and white
    if cv2.waitKey(5) & 0xFF == ord('c'):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]

                padding = 20
                x_min = max(0, int(x_min - padding))
                x_max = min(image.shape[1], int(x_max + padding))
                y_min = max(0, int(y_min - padding))
                y_max = min(image.shape[0], int(y_max + padding))

                hand_image = image[y_min:y_max, x_min:x_max]

                # Resize the hand image to a fixed size
                resized_hand_image = cv2.resize(hand_image, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)

                # Convert the resized hand image to grayscale
                gray_hand_image = cv2.cvtColor(resized_hand_image, cv2.COLOR_BGR2GRAY)

                # Save the grayscale hand image
                cv2.imwrite(f'hand_image_{image_counter}.jpg', gray_hand_image)
                print(f"Image saved as hand_image_{image_counter}.jpg")
                image_counter += 1

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and close the window
hands.close()
cap.release()
cv2.destroyAllWindows()
