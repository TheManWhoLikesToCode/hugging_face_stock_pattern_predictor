"""
Author: Jaydin 
Data: 03/16/2024
Description: This is a screen overlay that will allow the user to feed frames into the model and get the output
"""

# Imports
from ultralyticsplus import YOLO, render_result  # Loading and rendering results
import cv2  # Capturing and Image Processing
import numpy as np  # Array Manipulation
from mss import mss  # Screen Recording
from PIL import Image  # Image Processing
import pywinctl as pwc  # Window Capture and Selection

# Loading the model
stock_pattern_model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

# Model Parameters
stock_pattern_model.overrides['conf'] = 0.25  # NMS confidence threshold
stock_pattern_model.overrides['iou'] = 0.45  # NMS IoU threshold
stock_pattern_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
# maximum number of detections per image
stock_pattern_model.overrides['max_det'] = 1000

# Define a flag to indicate if the program is running
running = True

# * Step 0: Prompt the user to select a window


def prompt_window_selection():
    """
    Function to prompt the user to select a window

    Arguments:
        - None

    Output:
        - Window name

    """
    global running  # Flag to indicate if the program is running

    windows_titles = pwc.getAllTitles()

    print("Available Windows: ")
    for i, window_title in enumerate(windows_titles, start=0):
        print(f"{i+1}. {window_title}")

    # Prompt the user to select a window
    while running:
        window_number = input("Please select a window: ")

        try:
            window_number = int(window_number)
            if window_number > 0 and window_number <= len(windows_titles):
                window_name = windows_titles[window_number - 1]
                break
            else:
                print("Invalid window number. Please try again.")
        except ValueError:
            print("Invalid input. Please try again.")

    return window_name


# * Step 1: Capture a given window
def capture_window(window_name):
    """
    Function to capture a window given a window name

    Arguments:
        - Window name

    Output:
        - Numpy array of frames

    """
    # Define a flag to indicate if the program is running
    global running

    # Get the window based on the window name
    window = pwc.getWindowsWithTitle(window_name)

    if not window:
        print("Window not found. Please try again.")
        return

    # Get the window bounds
    left, top, right, bottom = window[0].rect
    width = right - left
    height = bottom - top

    # Define a bounding box for the screen recording
    screen_recording_bounds = {
        'left': left,
        'top': top,
        'width': width,
        'height': height,
    }

    # Create a screen recording object
    screen_recording_object = mss()

    # Create a event loop to capture the screen
    while running:
        # Capture the screen
        screen_capture = screen_recording_object.grab(screen_recording_bounds)

        screen_image = Image.frombytes(
            'RGB', (screen_capture.width, screen_capture.height), screen_capture.rgb)

        cv2.imshow('Window Capture', np.array(screen_image))

        # Turn the screen capture into a numpy array
        screen_numpy_array = np.array(screen_image)

        processed_frames = process_frames(screen_numpy_array)

        # Display the overlay
        cv2.imshow('Overlay', processed_frames)

        # Check if the user wants to exit the program
        if cv2.waitKey(33) & 0xFF in (
            ord('q'),
            27,
        ):
            running = False
            cv2.destroyAllWindows()
            break


# * Step 2: Process the frames
def process_frames(numpy_array):
    """
    Function to process frames

    Arguments:
        - Numpy array of frames

    Output:
        - Numpy array of frames with the overlay

    """
    # Get the output from the model
    results = stock_pattern_model(numpy_array)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Render the results
    # rendered_results = render_result(numpy_array, results)

    return annotated_frame


# * Step 4: Main Event Loop
def main():
    """
    Main event loop to run the program

    Arguments:
        - None

    Output:
        - None

    """

    # Capture the screen
    window_name = prompt_window_selection()
    capture_window(window_name)


if __name__ == "__main__":
    main()
