import paramiko
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.host_label = QLabel("Host:")
        self.host_input = QLineEdit()
        self.port_label = QLabel("Port:")
        self.port_input = QLineEdit()
        self.username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_server)
        
        layout = QVBoxLayout()
        layout.addWidget(self.host_label)
        layout.addWidget(self.host_input)
        layout.addWidget(self.port_label)
        layout.addWidget(self.port_input)
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.connect_button)
        
        self.setLayout(layout)
        
    def connect_to_server(self):
        # Get the user input
        host = self.host_input.text()
        #port = int(self.port_input.text())
        username = self.username_input.text()
        password = self.password_input.text()
        
        # Create an SSH client and connect to the server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=host, username=username, password=password)
        # Do something with the SSH client, e.g. run a command
        # stdin, stdout, stderr = ssh_client.exec_command("python")
        
        # stdin, stdout, stderr = ssh_client.exec_command("""print('hi')""")
        sftp_client = ssh_client.open_sftp()
        remote_file = sftp_client.open("/project/ag-moonraedler/MAtienza/Experiments/UNikon_30s_22-07-22/timelapse.nd2", "r")
        f = ND2Reader(remote_file)
        image = f.get_frame_2D()
        print('we have an image')
        plt.imshow(image)
        plt.show()
        
        
        
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
