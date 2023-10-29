import torch
import torch.nn as nn

class ImageEncoderModel(nn.Module):

  def __init__(self, image_size, latent_dim):
      """
      :param image_size: size of input image (W == H)
      :param latent_dim: number of channels of input image
      """
      super(ImageEncoderModel, self).__init__()

      self.image_size = image_size
      self.latent_dim = latent_dim
      self.flatten = nn.Flatten()
      self.ln1 = nn.Linear(in_features=self.image_size*self.image_size,
                           out_features= self.latent_dim)
      self.relu1 = nn.ReLU()

      self.ln2 = nn.Linear(in_features=self.latent_dim,
                           out_features=self.image_size*self.image_size)
      self.sigmoid = nn.Sigmoid()
      self.unflatten = nn.Unflatten(1, torch.Size([1, self.image_size, self.image_size]))

  def forward(self, x):
      """
      :param x: input image data
      :return: model output
      """
      x = self.flatten(x)
      x = self.ln1(x)
      x = self.relu1(x)

      x = self.ln2(x)
      x = self.sigmoid(x)
      x = self.unflatten(x)

      return x