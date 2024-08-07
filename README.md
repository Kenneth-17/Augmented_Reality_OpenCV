# Augmented Reality with OpenCV

This project demonstrates the implementation of a simple Augmented Reality (AR) application using Python and OpenCV. The application detects a predefined marker in a real-time video stream and overlays a custom image onto the detected marker using perspective transformations.

## Features

- **Real-Time Marker Detection**: Utilizes the SIFT feature detector to find and track a custom marker in the video stream.
- **Image Projection**: Projects a predefined image onto the detected marker using homography.
- **Robust Matching**: Employs FLANN-based matcher for efficient and robust matching of features.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What you need to install the software:

```bash
pip install opencv-python
pip install numpy
