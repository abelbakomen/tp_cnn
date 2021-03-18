import mnist
from conv import Conv2d

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv2d(8)
output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)
