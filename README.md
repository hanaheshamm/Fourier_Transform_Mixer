# Fourier Transform Mixer

## Overview

The Fourier Transform Mixer is an advanced desktop application designed for signal processing enthusiasts and professionals alike. It offers a unique platform to explore and understand the critical aspects of magnitude, phase, and frequency contributions in signals, with a special focus on 2D signals such as images. By enabling the manipulation and mixing of Fourier Transform (FT) components of grayscale images, the application serves as an educational tool to demonstrate the intricate ways in which these components influence the final composition of the signal.

## Features

### 1- Image Viewers
- **Multi-Viewport Support:** Facilitates the simultaneous viewing of four grayscale images, each within its own viewport.
- **Automatic Color Conversion:** Colored images are automatically converted to grayscale to maintain consistency in analysis.
- **Dynamic Size Adjustment:** Image sizes are automatically adjusted to match the dimensions of the smallest image, ensuring uniformity.
- **Flexible Display Options:** Users can select from original image, FT Magnitude, FT Phase, FT Real, and FT Imaginary components for display.
- **Intuitive Image Replacement:** Images can be easily replaced through a simple double-click action.

### 2- Output Ports
- **Dual Output Viewports:** Results from the mixing process can be displayed in one of two designated output viewports, with users having the ability to choose the target viewport.

### 3- Brightness and Contrast Adjustment
- **Interactive Image Enhancement:** Provides the capability for users to adjust the brightness and contrast of images and their components through direct mouse interaction.

### 4- Components Mixer
- **Customizable Component Weights:** Offers sliders to adjust the weights of the FT components from the input images, allowing for personalized visualization outcomes.

### 5- Regions Mixer
- **Targeted Frequency Selection:** Enables users to select specific regions of FT components (either low or high frequencies) for inclusion in the output.
- **Region Highlighting:** Selected regions are marked with semi-transparent overlays or hashing, with customizable size or percentage via sliders or resize handles.

### 6- Realtime Mixing
- **Progress Feedback:** Incorporates a progress bar to indicate the status of the mixing process, enhancing user engagement.
- **Operation Management:** Supports the cancellation and immediate restart of mixing operations in response to new user inputs.


## Screenshots
Below are screenshots from the application, showcasing its user interface and functionality:
![image](https://github.com/hanaheshamm/Fourier_Transform_Mixer/assets/115111861/02217043-1586-4a75-8a43-9a2ba4113ee4)

![image](https://github.com/hanaheshamm/Fourier_Transform_Mixer/assets/115111861/72bf6ad6-0801-4b3e-94d8-da540bfdca7a)

![image](https://github.com/hanaheshamm/Fourier_Transform_Mixer/assets/115111861/d6594daa-83f7-447f-b657-ee3ebce22f45)





## Usage Guidelines
- **Opening Images:** Navigate to the file menu or double-click on a viewport to open a grayscale image. The application will automatically convert colored images to grayscale.
- **Viewing FT Components:** Select the desired FT component for display through the dropdown menu associated with each image viewport.
- **Adjusting Brightness/Contrast:** Click and drag within an image viewport to adjust its brightness and contrast levels.
- **Mixing Components:** Utilize the sliders to fine-tune the weights of the FT components for the mixing process.
- **Selecting Regions:** Draw a rectangle on the FT display to choose a specific region for mixing. Adjust the size of the selected region using a slider or resize handles.

  
### Understanding FT Components
- **Magnitude:** Essential for defining the intensity of frequencies within the signal, directly impacting the contrast and sharpness of the reconstructed image.
- **Phase:** Critical for encoding the spatial information and structure of the image, pivotal in maintaining the image's original appearance.
- **Frequency Contributions:** Highlights the importance of both low and high frequencies in crafting the image's texture and edge details.

## Installation Instructions

```bash
# Clone the project repository
git clone [repository-link]
# Change directory to the project folder
cd FourierTransformMixer
# Install the required dependencies
pip install -r requirements.txt
# Execute the application
python main.py
