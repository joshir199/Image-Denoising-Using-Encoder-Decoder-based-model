# Image-Denoising-Using-Encoder-Decoder-based-model
Understanding and Implementing the Image Reconstruction and Image denoising techniques in Computer Vision Area


# Image Reconstruction

The autoencoder based model is used to reconstruct the image from its latent representation. The model is trained on the standard Fashion MNIST dataset with image size of 28x28 and one channel.

![](https://github.com/joshir199/Image-Denoising-Using-Encoder-Decoder-based-model/blob/main/images/real_image.png)

------------------>  training image sample


Each training image has 28x28 = 784 features. The model will use the latent dimension of 64 to compress the image representation and using decoder block it will recreate the image similar to original image with minimum reconstruction error.

![](https://github.com/joshir199/Image-Denoising-Using-Encoder-Decoder-based-model/blob/main/output/autoencoder_model_summary.png)

---------> Autoencoder model for image reconstruction


------------------------------
For Training, following scipt can be run:
```bash
python train_autoencoder.py --learning_rate 0.001 --train
```

![](https://github.com/joshir199/Image-Denoising-Using-Encoder-Decoder-based-model/blob/main/output/autoencoder_loss.png)

---------------> training MSE Loss graph 

After training the model, it can be used to reconstruct the input image like below. 

![](https://github.com/joshir199/Image-Denoising-Using-Encoder-Decoder-based-model/blob/main/output/autoencoder_predicted_image.png)

-----------------> Image reconstructed using model


Use Cases:  This Image Reconstruction image can be used to compress the image, detect anamoly in the image or any other data by comparing the reconstructed result.

