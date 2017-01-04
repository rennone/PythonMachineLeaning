from DataSet.mnist import load_mnist
from PIL import Image
import numpy as np
from DataSet.img_show import img_show


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]

print(label)

img = img.reshape(28, 28)
img_show(img)