from collections.abc import Iterable
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.model import EllipseI
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from getpass import getpass
import keyring
from tqdm import tqdm
import time

class Omero:

    def __init__(self, hostname, username):
        """
        Connect to an OMERO server
        :param hostname: Host name
        :param username: User
        :param password: Password
        :return: Connected BlitzGateway
        """
        #password = keyring.get_password('omero', username)
        if True:
            print('I will now get your password')
            password = getpass(f'Password for {username}:')

        conn = BlitzGateway(username, password,
                        host=hostname, secure=True)
        conn.connect()
        conn.c.enableKeepAlive(60)

        self.conn = conn

    
    def disconnect(self):
        """
        Disconnect from an OMERO server
        :param conn: The BlitzGateway
        """
        self.conn.close()

    
    def get_dataset(self, dataset_id):

        return self.conn.getObject('Dataset', dataset_id)

    
    def get_image_shape(self, image_id):


        image = self.conn.getObject('Image', image_id)

        shape = image.getSizeT(), image.getSizeY(), image.getSizeX(), image.getSizeC()

        return shape

        
    def get_np(self, image_id, frames, channel=0):

        image = self.conn.getObject('Image', image_id)

        if not isinstance(frames, Iterable):

            return image.get_PrimaryPixels().get_Plane(frames, channel)

        if isinstance(frames, str):

            frames = np.arange(0, image.getSizeT()) 

        if isinstance(frames, Iterable):

            image_0 = image.getPrimaryPixels().getPlane(0,channel, frames[0])
            shape = frames.size, image.getSizeY(), image.getSizeX(), image.getSizeC()
            dtype = image_0.dtype

            # zct_list = []
            # for t in frames:
            #     zct_list.append((0,channel, t))
            # t_0 = time.time()
            # print('going for it')
            # pixels = np.zeros(shape[:-1], dtype)
            # pixels_gen = image.getPrimaryPixels().getPlanes(zct_list)
            # print(time.time()-t_0)
            # for i in range(pixels.shape[0]):
            #     pixels[i] = pixels_gen[i]
            # return pixels_gen

            pixels = np.zeros(shape[:-1], dtype)
            pixels[0] = image_0

            zct_list = []

            for t in frames:
                zct_list.append((0, channel, t))

            generator = image.getPrimaryPixels().getPlanes(zct_list)
            
            print('Downloading from Omero...')
            for i in tqdm(np.arange(1, frames.size)):

                 pixels[i] = generator.__next__()
    
            
            
            
            # print('Downloading from Omero...')
            # for i in tqdm(np.arange(1, frames.size)):

            #     pixels[i] = image.getPrimaryPixels().getPlane(0, channel, i)
    
        return pixels

        




    
    