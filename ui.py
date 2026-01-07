import sys
import os

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")  # Use Qt backend for embedding
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import image

# -------------------------------------------------------------------
# PyQt Application
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Character Extraction Viewer")
        self.resize(900, 700)

        # State for current sequence of plots
        self.figures = []            # list[Figure]
        self.current_plot_index = -1 # index in self.figures
        self.canvas = None           # current FigureCanvas

        # ---- Layout ----
        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        # Info label
        self.info_label = QLabel("Select an image to start.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.info_label)

        # Open button
        self.open_button = QPushButton("Open Image...")
        self.open_button.clicked.connect(self.open_image)
        self.main_layout.addWidget(self.open_button)

        # Next/Home button
        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.show_next_plot)
        self.main_layout.addWidget(self.next_button)

        # Area for a single matplotlib canvas
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)
        self.main_layout.addWidget(self.plot_container, stretch=1)

        self.setCentralWidget(container)

    # ---------------- Core actions ----------------
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "File does not exist.")
            return

        self.start_plot_sequence(file_path)

    def start_plot_sequence(self, image_path):
        """
        Call extract_characters_from_image and start showing
        the returned figures one by one in a single window.
        """
        # Clear any previous sequence
        self.clear_current_canvas()
        self.figures = []
        self.current_plot_index = -1

        result = image.extract_characters_from_image(image_path)

        if len(result.plots) == 0:
            QMessageBox.information(self, "No Plots", "No figures returned.")
            self.return_to_main_menu()
            return

        self.figures = result.plots

        self.current_plot_index = 0
        total = len(self.figures)
        self.info_label.setText(
            f"Showing plot 1 of {total} for {os.path.basename(image_path)}"
        )

        # Enable button if there is at least one plot
        self.next_button.setEnabled(True)

        # Show first plot (this will also set button text to Next/Home)
        self.show_plot_at_index(self.current_plot_index)

    def show_next_plot(self):
        """
        If not on last plot: go to next plot.
        If on last plot (button shows 'Home'): go back to main menu.
        """
        if not self.figures:
            return

        # If we are on the last plot, this acts as "Home"
        if self.current_plot_index >= len(self.figures) - 1:
            self.return_to_main_menu()
            return

        # Otherwise, go to next plot
        self.current_plot_index += 1
        self.show_plot_at_index(self.current_plot_index)

    def show_plot_at_index(self, idx: int):
        """
        Replace the current canvas with the figure at index idx.
        Also update the button text: 'Next' or 'Home'.
        """
        self.clear_current_canvas()

        fig = self.figures[idx]
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(
            self.canvas.sizePolicy().Expanding,
            self.canvas.sizePolicy().Expanding,
        )
        self.plot_layout.addWidget(self.canvas)

        total = len(self.figures)
        self.info_label.setText(f"Showing plot {idx + 1} of {total}")

        # If this is the last plot, show 'Home'; otherwise 'Next'
        if idx == total - 1:
            self.next_button.setText("Home")
        else:
            self.next_button.setText("Next")

    def return_to_main_menu(self):
        """
        Reset to initial 'main menu' state.
        """
        self.clear_current_canvas()
        self.figures = []
        self.current_plot_index = -1
        self.next_button.setEnabled(False)
        self.next_button.setText("Next")  # reset label
        self.info_label.setText("Select an image to start.")

    # ---------------- Helpers ----------------
    def clear_current_canvas(self):
        """
        Remove the currently displayed canvas (if any).
        """
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()