import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from nd2reader import ND2Reader
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QSpacerItem, QFileDialog, QLabel)

from skimage import io

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
	
        # Create the central widget and set it as the main window's central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a button to trigger the file browser
        self.nd2_file_button = QPushButton("Select nd2 image")
        self.nd2_file_button.clicked.connect(self.openFileBrowser)

        # Open the file browser at the beginning
        self.image_file,_ = QFileDialog.getOpenFileName(self,"Select nd2 image", "","ND2 Files (*.nd2)")
        #self.openFileBrowser()
        self.f = ND2Reader(self.image_file)

        # Create the figure and the canvas for the image
        self.figure_image, self.ax_image = plt.subplots(constrained_layout=True)
        self.image = self.ax_image.imshow(np.zeros((100, 100)), clim=(0, 255))
        self.ax_image.set_xticklabels([])
        self.ax_image.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.canvas_image = FigureCanvas(self.figure_image)

        toolbar = NavigationToolbar(self.canvas_image, self)


        screen_height = QApplication.primaryScreen().geometry().height()

        self.canvas_image.setMinimumWidth(int(screen_height*0.7))


        # Create the figure and the canvas for the plot
        self.figure_plot, self.ax_plot = plt.subplots(constrained_layout=True)
        x, y = [1,2], [2, 3]
        self.plot, = self.ax_plot.plot(x, y)
        self.canvas_plot = FigureCanvas(self.figure_plot)

        # Create the sliders and buttons
        self.slider1 = QSlider(Qt.Horizontal)
        self.t_slider = QSlider(Qt.Horizontal)
        #self.t_slider.valueChanged.connect(self.update_nd2)
        self.button1 = QPushButton('Update Image', self)
        self.button2 = QPushButton('Update Plot', self)

        # Create a button to trigger the file browser
        self.nd2_file_button = QPushButton("Select nd2 image")
        self.nd2_file_button.clicked.connect(self.openFileBrowser)
        
        # Create a vertical layout for the sliders and buttons
        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.slider1)
        #vbox_left.addWidget(self.slider2)
        vbox_left.addWidget(self.button1)
        vbox_left.addWidget(self.button2)
        vbox_left.addWidget(self.nd2_file_button)

        #Veritcal box for image and toolbar and t slider
        vbox_image = QVBoxLayout()
        vbox_image.addWidget(self.canvas_image)
        vbox_image.addWidget(toolbar)
        vbox_image.addWidget(self.t_slider)

        # Create a horizontal layout to split the main window into two columns
        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_image, stretch=1)
        hbox.addWidget(self.canvas_plot, stretch=1)
        #hbox.addWidget(spacer, stretch=1)
        #hbox.addStretch(1)

        # Use a spacer to stretch the right column horizontally
        hbox.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Add the horizontal layout to the central widget
        central_widget.setLayout(hbox)

        # Connect the buttons to their respective functions
        self.button1.clicked.connect(self.update_nd2)
        self.button2.clicked.connect(self.update_plot)
    
        
    def update_image(self):
        # Get the values of the sliders
        if 'image_file' in dir(self):
            print('hi')
            image = io.imread(self.image_file)
            self.image.set_data(image)
            self.canvas_image.draw()
            return
            
        slider1_value = self.slider2.value()
        #slider2_value = self.slider2.value()

        # Generate new image data based on the slider values
        new_data = generate_image_data(slider1_value)
        # Update the image data
        self.image.set_data(new_data)

        # Redraw the canvas
        self.canvas_image.draw()
    
    def update_nd2(self):

        #vmin, vmax = self.clip.value
        #clip=self.clip.value
        #t = self.t_slider.value
        #c =self.c.value
        #v=self.v.value
        #image = self.f.get_frame_2D(v=v,c=c,t=t)
               
        #self.im.set_data(image)
        #lanes = g.get_frame_2D(v=v)
        #self.im.set_clim([vmin, vmax])
        #self.fig.canvas.draw()
        
        # if v!=self.oldv:
        #     self.cyto_locator=None
        #     self.update_lanes()
        
        # if self.view_nuclei.value:
        #     if v!=self.oldv:
        #         self.load_df(self.db_path, v)
        #     self.update_tracks()
            
            
        # if self.view_cellpose.value:
        #     if v!=self.oldv:
        #         self.load_masks(self.outpath, v)
    
        cyto = self.f.get_frame_2D(t=self.t_slider.value())
        self.image.set_data(cyto)

        # Redraw the canvas
        self.canvas_image.draw()
        #self.tmarker.set_xdata(t)
        
        #self.oldv=v

    def update_plot(self):
        # Get the values of the sliders
        slider1_value = self.slider1.value()
        slider2_value = self.slider2.value()

        # Generate new plot data based on the slider values
        x, y = generate_plot_data(slider1_value, slider2_value)

        # Update the plot data
        self.ax.lines[0].set_xdata(x)
        self.ax.lines[0].set_ydata(y)

        # Update the limits of the plot
        self.ax.relim()
        self.ax.autoscale()

        # Redraw the canvas
        self.canvas.draw()
    
    # Create a function to open the file browser when a button is clicked
    def openFileBrowser(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            self.image_file = fileName
            pass#self.label.setText(fileName)

def generate_image_data(slider1_value):
    
    image = np.ones((100, 100))*slider1_value
    #image = io.imread(self.image_file)
    return image

   
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
