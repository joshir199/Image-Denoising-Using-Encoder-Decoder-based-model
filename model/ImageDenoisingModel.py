import torch.nn as nn


class ImageDenoisingModel(nn.Module):

  def __init__(self, image_size, channels=1):
      """

      :param image_size: size of input image (W == H)
      :param channels: number of channels of input image
      """
      super(ImageDenoisingModel, self).__init__()

      self.image_size = image_size
      self.channels = channels
      self.conv2d_1 = nn.Conv2d(in_channels=self.channels, out_channels=32, kernel_size=3, stride=2, padding=1)
      self.relu1 = nn.ReLU()
      self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
      self.relu2 = nn.ReLU()

      # Used output_padding to match output size (equivalent to padding='same' of tensorflow)
      self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)
      self.relu3 = nn.ReLU()
      self.conv_transpose_2 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)
      self.relu4 = nn.ReLU()
      self.conv2d_3 = nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
      self.sigmoid = nn.Sigmoid()



  def forward(self, x):
      """
      :param x: input image data
      :return: model output
      """

      # encoder part
      x = self.conv2d_1(x)
      x = self.relu1(x)
      x = self.conv2d_2(x)
      x = self.relu2(x)

      # decoder part
      x = self.conv_transpose_1(x)
      x = self.relu3(x)
      x = self.conv_transpose_2(x)
      x = self.relu4(x)
      x = self.conv2d_3(x)
      x = self.sigmoid(x)

      return x