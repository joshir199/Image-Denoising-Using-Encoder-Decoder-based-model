import matplotlib.pyplot as plt
from torchsummary import summary
import torch


class Utils:

    @staticmethod
    def showData(data_point, image_size):
        plt.imshow(data_point[0].numpy().reshape(image_size, image_size), cmap='gray')
        plt.title('y = ' + str(data_point[1]))
        plt.show()


    @staticmethod
    def saveModelSummary(model, channels, image_size):
        summary(model, input_size=(channels, image_size, image_size))

    @staticmethod
    def showNoisyData(data_point, image_size, noise_factor):
        data = data_point[0] + torch.randn_like(data_point[0]) * noise_factor
        data = torch.clip(data, min=0.0, max=1.0)
        plt.imshow(data.numpy().reshape(image_size, image_size), cmap='gray')
        plt.show()
        return data

    @staticmethod
    def showDataPredicted(data_point, image_size):
        plt.imshow(data_point.numpy().reshape(image_size, image_size), cmap='gray')
        plt.show()
