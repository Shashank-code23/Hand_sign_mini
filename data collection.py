import os
import cv2

# Define the data directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes (A to Z)
number_of_classes = 26

# Define the dataset size per class
dataset_size = 100

# Initialize the camera
cap = cv2.VideoCapture(0)

# Loop through each class (A to Z)
for j in range(number_of_classes):
    # Create a directory for each class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, chr(65 + j))):
        os.makedirs(os.path.join(DATA_DIR, chr(65 + j)))

    # Prompt the user to get ready for capturing the current letter's sign
    print('Collecting data for letter "{}"'.format(chr(65 + j)))

    # Wait for the user to press 'Q' to start capturing
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Get ready for letter "{}". Press "Q" ! :)'.format(chr(65 + j)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture and save images for the current letter
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, 'Capturing letter "{}"'.format(chr(65 + j)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, chr(65 + j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
