import sys
import cv2
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
import numpy as np
from numpy.fft import fft2, fftshift
# from Classes import loadUiType, path
from PyQt5.uic import loadUiType
from os import path
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QTimer


from PIL import Image

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "MainWindow_final.ui"))




class ImageViewer:
    def __init__(self, label_widget, component_widget):
        self.label_widget = label_widget
        self.component_widget = component_widget
        self.copy_image=None
        # self.output_widget = output_widget
        self.image = None
        self.width = 0
        self.height = 0
        self.original_image = None
        self.ft_magnitude_component = None
        self.ft_phase_component = None
        self.ft_real_component = None
        self.ft_imaginary_component = None
        # self.main_app = MainApp
        self.main_app = None
        # self.setup_label_widget()
        self.mouse_pressed = False
        self.last_mouse_position = None

        # self.main_app.rectangle_slider.valueChanged.connect(self.updateRectangle)

    def setup_label_widget(self):
        if self.label_widget is not None:
            # Enable mouse tracking on the label widget
            self.label_widget.setMouseTracking(True)
            # Connect the mouse move event to a handler function
            self.label_widget.mousePressEvent = self.mousePressEvent
            self.label_widget.mouseReleaseEvent = self.mouseReleaseEvent
            self.label_widget.mouseMoveEvent = self.mouseMoveEvent





    def show_image(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label_widget.setPixmap(pixmap)
        self.label_widget.setScaledContents(True)

    def show_component(self, ft_component , slider_value):
        # pixmap = QPixmap.fromImage(ft_component)
        # self.component_widget.setPixmap(pixmap)
        # self.component_widget.setScaledContents(True)


        self.currentComponent = ft_component

        # Create a new pixmap for the component image
        component_pixmap = QPixmap.fromImage(ft_component)
        component_pixmap = component_pixmap.scaledToWidth(self.component_widget.width())
        self.component_widget.setPixmap(component_pixmap)
        self.component_widget.setScaledContents(True)

        # Draw a rectangle on a separate pixmap
        rect_pixmap = QPixmap(component_pixmap.size())
        rect_pixmap.fill(Qt.transparent)
        rect_painter = QPainter(rect_pixmap)

        pen = QPen(QColor(255, 0, 0))  # Red color for the rectangle
        pen.setWidth(2)  # Set the width of the rectangle border
        rect_painter.setPen(pen)

        # Calculate the center coordinates for the rectangle
        rect_width = 50 + slider_value//2  # Adjust the rectangle size based on the slider value
        rect_height = 50 + slider_value // 2  # Adjust the rectangle size based on the slider value
        rect_x = (component_pixmap.width() - rect_width) // 2
        rect_y = (component_pixmap.height() - rect_height) // 2

        rect = QRect(rect_x, rect_y, rect_width, rect_height)
        rect.moveCenter(component_pixmap.rect().center())

        # Draw the rectangle on the pixmap
        rect_painter.drawPixmap(0, 0, component_pixmap)
        rect_painter.drawRect(rect)
        rect_painter.end()

        # Create a new pixmap to overlay the rectangle on the component image
        overlay_pixmap = QPixmap(component_pixmap.size())
        overlay_pixmap.fill(Qt.transparent)
        overlay_painter = QPainter(overlay_pixmap)

        # Draw the component image
        overlay_painter.drawPixmap(0, 0, component_pixmap)

        # Draw the rectangle on top of the component image
        overlay_painter.drawPixmap(0, 0, rect_pixmap)

        overlay_painter.end()

        # Set the overlaid pixmap on the component widget
        self.component_widget.setPixmap(overlay_pixmap)

    def updateAndShowComponent(self, slider_value):
            if self.main_app is not None:
                self.show_component(self.currentComponent, slider_value)



    # def apply_fourier_transform_afterroi(self,arr_roi,index):
    #     if index == 0:
    #         self.ft_real_component = arr_roi
    #     elif index == 1:
    #         self.ft_imaginary_component = arr_roi
    #     elif index == 2:
    #         self.ft_magnitude_component = arr_roi
    #     elif index == 3:
    #         self.ft_phase_component_component = arr_roi
    def apply_fourier_transform_afterroi(self, arr_roi, index):
        components = {
            0: "ft_real_component",
            1: "ft_imaginary_component",
            2: "ft_magnitude_component",
            3: "ft_phase_component"
        }

        if index in components:
            setattr(self, components[index], arr_roi)

    def convert_to_grayscale(self):
        self.image = self.image.convertToFormat(QImage.Format_Grayscale8)

    def apply_fourier_transform(self):
        width = self.image.width()
        height = self.image.height()
        bytes_per_line = self.image.bytesPerLine()
        ptr = self.image.bits()
        ptr.setsize(height * bytes_per_line)
        arr = np.array(ptr).reshape((height, width))
        f_transform = fft2(arr)
        f_transform_shifted = fftshift(f_transform)
        self.ft_magnitude_component = np.abs(f_transform_shifted)
        # self.ft_magnitude_component = np.log(self.ft_magnitude_component + 1)
        self.ft_phase_component = np.angle(f_transform_shifted)
        self.ft_real_component = np.real(f_transform_shifted)
        self.ft_imaginary_component = np.imag(f_transform_shifted)


    def create_image_from_component(self, component):
        # if component == self.ft_magnitude_component:
        #     component = np.log(component + 1)
        # self.ft_magnitude_component = np.log(self.ft_magnitude_component + 1)

        component = (component / np.max(component) * 255).astype(np.uint8)
        bytes_per_line = component.shape[1]
        q_image = QImage(component.data.tobytes(), self.width, self.height, bytes_per_line, QImage.Format_Grayscale8)
        return q_image

    def load_image_class(self, image_path):
        self.image = cv2.imread(image_path, 0)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        # Adjust brightness and contrast
        # Convert the adjusted OpenCV image back to QImage
        # new_image = cv2.convertScaleAbs(self.image, alpha=1, beta=0)
        self.image = QImage(self.image, self.width, self.height, self.width, QImage.Format_Grayscale8)
        # Display the updated image
        self.show_image(self.image)


    def adjust_brightness_and_contrast(self, alpha=1.0, beta=0):

        alpha = max(0.1, min(alpha, 3.0))
        beta = max(-100, min(beta, 100))

        #take a copy from image
        width = self.image.width()
        height = self.image.height()
        bytes_per_line = self.image.bytesPerLine()
        ptr = self.image.bits()
        ptr.setsize(height * bytes_per_line)
        arr_2 = np.array(ptr).reshape((height, width))
        self.copy_image = arr_2.copy()

        # Adjust brightness and contrast
        adjusted_image = cv2.convertScaleAbs(arr_2, alpha=alpha, beta=beta)

        # # Convert the adjusted OpenCV image back to QImage
        adjusted_image = QImage(adjusted_image.data, self.width, self.height, self.width, QImage.Format_Grayscale8)
        #
        # # Display the updated image
        self.show_image(adjusted_image)

    def mousePressEvent(self, event):
        # On mouse press, record the position and set the flag
        self.last_mouse_position = event.pos()
        print(self.last_mouse_position)
        self.mouse_pressed = True
        print("clicked")
        event.accept()



    def mouseReleaseEvent(self, event):
        # On mouse release, reset the last position and the flag
        self.last_mouse_position = None
        self.mouse_pressed = False
        print("released")
        event.accept()




    def mouseMoveEvent(self, event):
        # On mouse move, if the mouse is pressed, adjust brightness/contrast
        if self.mouse_pressed and self.last_mouse_position:
            # Calculate the difference
            current_position = event.pos()
            dx = current_position.x() - self.last_mouse_position.x()
            dy = current_position.y() - self.last_mouse_position.y()

            # Map the mouse movement to contrast and brightness changes
            # These mappings are arbitrary and you can adjust them as needed
            alpha = 1.0 + (dx / 50.0)  # Adjust contrast based on x movement
            beta = dy * 2 # Adjust brightness based on y movement

            self.adjust_brightness_and_contrast(alpha, beta)

            # Update the last position for the next movement
            self.last_mouse_position = current_position
        event.accept()

    def keyPressEvent(self, event):
        # Check if 'R' key is pressed
        if event.key() == Qt.Key_R:
            # Reset the image to the original
            self.reset_to_original_image()
            print("reset")

            # Update the display to show the original image
            self.show_image(self.image)

        # Accept the event or pass it to the base class
        event.accept()

    def reset_to_original_image(self):
        # Assuming you have a method to reset your image to the original
        # This method would set the image being displayed to the original image
        self.image = self.copy_image.copy()




    def load_component_image(self, index):
        if self.image is not None:
            self.apply_fourier_transform()
            # ft_mag_image = self.create_image_from_component(self.ft_magnitude_component)
            ft_mag_image = self.create_image_from_component( np.log(self.ft_magnitude_component + 1))
            ft_phase_image = self.create_image_from_component(self.ft_phase_component)
            ft_real_image = self.create_image_from_component(self.ft_real_component)
            ft_imaginary_image = self.create_image_from_component(self.ft_imaginary_component)

            sliderValue= self.main_app.rectangle_slider.value()
            component_mapping = {
                0: ft_mag_image,
                1: ft_phase_image,
                2: ft_real_image,
                3: ft_imaginary_image
            }
            self.show_component(component_mapping.get(index, ft_mag_image), sliderValue)




class ImageMixer:
    def __init__(self, image_viewer_1, image_viewer_2, image_viewer_3, image_viewer_4):
        self.image_viewer_1 = image_viewer_1
        self.image_viewer_2 = image_viewer_2
        self.image_viewer_3 = image_viewer_3
        self.image_viewer_4 = image_viewer_4
        self.main_app = None
        self.total_steps = 10  # Adjust this based on the actual number of steps in your process

        # Create a QTimer to periodically update the progress bar
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_progress_bar)
        self.current_step = 0

    def apply_mixer(self):
        selected_index = self.main_app.mixer_output_comboBox.currentIndex()
        if selected_index == 0:
            self.update_mixer( self.main_app.output1_widget)
        elif selected_index == 1:
            self.update_mixer(self.main_app.output2_widget)
        else:
            self.update_mixer(self.main_app.output1_widget)



    def get_weighted_component(self, viewer, component_type, weight):
        component = {
            "real": viewer.ft_real_component,
            "imaginary": viewer.ft_imaginary_component,
            "magnitude": viewer.ft_magnitude_component,
            "phase": viewer.ft_phase_component,
        }.get(component_type.lower(), 0)
        return weight * component

    def update_progress(self, progress):
        self.main_app.progressBar.setValue(progress)

    def update_mixer(self, output_widget):

        self.update_progress(0)
        rectangleSlider = self.main_app.rectangle_slider.value()

        # Initial values for the mixed components
        mixed_real_component = 0
        mixed_imaginary_component = 0
        mixed_magnitude_component = 0
        mixed_phase_component = 0

        # Retrieve the weights and choices from the UI
        weights = [
            self.main_app.mixer_component1_slider.value() / 100,
            self.main_app.mixer_component2_slider.value() / 100,
            self.main_app.mixer_component3_slider.value() / 100,
            self.main_app.mixer_component4_slider.value() / 100,
        ]

        choice_images = [
            self.main_app.mixed_component1_comboBox.currentText().lower(),
            self.main_app.mixed_component2_comboBox.currentText().lower(),
            self.main_app.mixed_component3_comboBox.currentText().lower(),
            self.main_app.mixed_component4_comboBox.currentText().lower(),
        ]

        image_viewers = [self.image_viewer_1, self.image_viewer_2, self.image_viewer_3, self.image_viewer_4]

        # Process each viewer and accumulate the weighted components
        # for i, viewer in enumerate(image_viewers):
        #     viewer.apply_fourier_transform()
        #     viewer.apply_fourier_transform_afterroi(
        #         self.main_app.select_region(viewer.ft_real_component, rectangleSlider), 0
        #     )
        #     viewer.apply_fourier_transform_afterroi(
        #         self.main_app.select_region(viewer.ft_imaginary_component, rectangleSlider), 1
        #     )
        #     viewer.apply_fourier_transform_afterroi(
        #         self.main_app.select_region(viewer.ft_magnitude_component,rectangleSlider), 2
        #     )
        #     viewer.apply_fourier_transform_afterroi(
        #         self.main_app.select_region(viewer.ft_phase_component, rectangleSlider), 3
        #     )
        component_mapping = {
            'real': 0,
            'imaginary': 1,
            'magnitude': 2,
            'phase': 3
        }

        for i, viewer in enumerate(image_viewers):
            for component_name, index in component_mapping.items():
                component = getattr(viewer, f"ft_{component_name}_component")
                region = self.main_app.select_region(component, rectangleSlider)
                viewer.apply_fourier_transform_afterroi(region, index)

            if choice_images[i] == "real":
                mixed_real_component += weights[i] * viewer.ft_real_component
            elif choice_images[i] == "imaginary":
                mixed_imaginary_component += weights[i] * viewer.ft_imaginary_component
            elif choice_images[i] == "magnitude":
                mixed_magnitude_component += weights[i] * viewer.ft_magnitude_component
            elif choice_images[i] == "phase":
                mixed_phase_component += weights[i] * viewer.ft_phase_component

            self.update_progress((i + 1) * 25)

        # Determine the mixed_ft_component based on the first choice
        if choice_images[0] in ["real", "imaginary"]:
            mixed_ft_component = mixed_real_component + 1j * mixed_imaginary_component
        elif choice_images[0] in ["magnitude", "phase"]:
            mixed_ft_component = mixed_magnitude_component * np.exp(1j * mixed_phase_component)

        # Compute the inverse Fourier transform and normalize the image
        # normalized_mixed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(mixed_ft_component)))
        normalized_mixed_image = np.fft.ifft2(np.fft.ifftshift(mixed_ft_component))
        normalized_mixed_image = np.abs(normalized_mixed_image)
        normalized_mixed_image = ((normalized_mixed_image - normalized_mixed_image.min()) /
                                  (normalized_mixed_image.max() - normalized_mixed_image.min())) * 255.0
        normalized_mixed_image = normalized_mixed_image.astype(np.uint8)

        # Convert the normalized image to QPixmap and display it
        q_image = QImage(
            normalized_mixed_image.data.tobytes(),
            normalized_mixed_image.shape[1],
            normalized_mixed_image.shape[0],
            normalized_mixed_image.shape[1],
            QImage.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(q_image)
        output_widget.setPixmap(pixmap)
        output_widget.setScaledContents(True)
        self.update_progress(100)
    def update_progress_bar(self):
        # Calculate the current progress based on the current step and total steps
        progress = int((self.current_step / self.total_steps) * 100)

        # Update the progress bar
        self.main_app.progressBar.setValue(progress)




class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.image_viewer_1 = ImageViewer(self.image1_widget, self.image1_component_widget)
        self.image_viewer_2 = ImageViewer(self.image2_widget, self.image2_component_widget)
        self.image_viewer_3 = ImageViewer(self.image3_widget, self.image3_component_widget)
        self.image_viewer_4 = ImageViewer(self.image4_widget, self.image4_component_widget)
        self.image_viewer_1.main_app =self
        self.image_viewer_2.main_app =self
        self.image_viewer_3.main_app =self
        self.image_viewer_4.main_app =self

        self.image_mixer_instance = ImageMixer(self.image_viewer_1, self.image_viewer_2, self.image_viewer_3, self.image_viewer_4)
        self.image_mixer_instance.main_app = self


        self.register_event_handlers()

        self.rectangle_slider.setRange(0,255)
        # self.rectangle_slider.valueChanged.connect(self.image_viewer_1.updateRectangle)

        self.rectangle_slider.valueChanged.connect(self.image_viewer_1.updateAndShowComponent)
        self.rectangle_slider.valueChanged.connect(self.image_viewer_2.updateAndShowComponent)
        self.rectangle_slider.valueChanged.connect(self.image_viewer_3.updateAndShowComponent)
        self.rectangle_slider.valueChanged.connect(self.image_viewer_4.updateAndShowComponent)

        self.mixer_component1_slider.setValue(0)
        self.mixer_component2_slider.setValue(0)
        self.mixer_component3_slider.setValue(0)
        self.mixer_component4_slider.setValue(0)
        self.progressBar.setValue(0)

    def register_event_handlers(self):
        self.insert_image1_button.clicked.connect(lambda: self.load_image(self.image_viewer_1))
        self.insert_image2_button.clicked.connect(lambda: self.load_image(self.image_viewer_2))
        self.insert_image3_button.clicked.connect(lambda: self.load_image(self.image_viewer_3))
        self.insert_image4_button.clicked.connect(lambda: self.load_image(self.image_viewer_4))

        self.image1_component_comboBox.currentIndexChanged.connect(lambda index: self.load_component_image_handler(index, self.image_viewer_1))
        self.image2_component_comboBox.currentIndexChanged.connect(lambda index: self.load_component_image_handler(index, self.image_viewer_2))
        self.image3_component_comboBox.currentIndexChanged.connect(lambda index: self.load_component_image_handler(index, self.image_viewer_3))
        self.image4_component_comboBox.currentIndexChanged.connect(lambda index: self.load_component_image_handler(index, self.image_viewer_4))

        self.image1_widget.mouseDoubleClickEvent = lambda event: self.load_image(self.image_viewer_1)
        self.image2_widget.mouseDoubleClickEvent = lambda event: self.load_image(self.image_viewer_2)
        self.image3_widget.mouseDoubleClickEvent = lambda event: self.load_image(self.image_viewer_3)
        self.image4_widget.mouseDoubleClickEvent = lambda event: self.load_image(self.image_viewer_4)
        self.image1_widget.mousePressEvent = lambda event: self.image_viewer_1.mousePressEvent(event)
        self.image1_widget.mouseReleaseEvent = lambda event: self.image_viewer_1.mouseReleaseEvent(event)
        self.image2_widget.mousePressEvent = lambda event: self.image_viewer_2.mousePressEvent(event)
        self.image2_widget.mouseReleaseEvent = lambda event: self.image_viewer_2.mouseReleaseEvent(event)
        self.image3_widget.mousePressEvent = lambda event: self.image_viewer_3.mousePressEvent(event)
        self.image3_widget.mouseReleaseEvent = lambda event: self.image_viewer_3.mouseReleaseEvent(event)
        self.image4_widget.mousePressEvent = lambda event: self.image_viewer_4.mousePressEvent(event)
        self.image4_widget.mouseReleaseEvent = lambda event: self.image_viewer_4.mouseReleaseEvent(event)





        #apply mouse right click

        # self.apply_mixer_button.clicked.connect(self.update_mixer)
        # self.apply_mixer_button.clicked.connect(self.apply_mixer)
        self.apply_mixer_button.clicked.connect(self.image_mixer_instance.apply_mixer)





    def load_image(self , viwerObject):
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if imagePath:
            print("Selected image path:", imagePath)  # Debugging statement
            # self.image_viewer.load_image_class(imagePath)
            viwerObject.load_image_class(imagePath)
            viwerObject.load_component_image(0)
            viwerObject.setup_label_widget()

    def load_component_image_handler(self, index , viwerObject):
        # self.image_viewer_1.load_component_image(index)
        viwerObject.load_component_image(index)


    def select_region(self, image_arr, slider_value):
        selected_index = self.region_comboBox.currentIndex()
        options = {
            0: (True, 255),
            1: (True, slider_value),
            2: (False, slider_value),
        }
        inner, slider_value = options.get(selected_index, (True, 255))

        if inner:
            mask = np.zeros(image_arr.shape, dtype=bool)  # Use a boolean mask
        else:
            mask = np.ones(image_arr.shape, dtype=bool)  # Use a boolean mask

        height, width = image_arr.shape
        half_height = slider_value
        half_width = slider_value

        top_left_y = max(0, height // 2 - half_height)
        top_left_x = max(0, width // 2 - half_width)
        bottom_right_y = min(height, height // 2 + half_height)
        bottom_right_x = min(width, width // 2 + half_width)

        if inner:
            mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = True
        else:
            mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = False

        result_arr = np.zeros_like(image_arr)  # Create an array filled with zeros
        result_arr[mask] = image_arr[mask]     # Copy the values inside the rectangle

        return result_arr




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
