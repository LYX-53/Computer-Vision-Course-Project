import numpy as np
import matplotlib.pyplot as plt

# 使用 NumPy 加载 NPY 文件
depthmap = np.load('mnist_test_seq .npy')

# 使用 Matplotlib 显示图像
plt.imshow(depthmap)
plt.show()

# 将图像保存为 JPG 格式
plt.savefig('depthmap.jpg')
