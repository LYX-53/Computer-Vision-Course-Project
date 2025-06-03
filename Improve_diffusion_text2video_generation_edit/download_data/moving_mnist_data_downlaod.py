import os
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py
from moviepy.editor import ImageSequenceClip

def download_moving_mnist(download_dir: str) -> None:
    """
    Downloads the Moving MNIST dataset.

    :param download_dir: Directory where the dataset will be saved.
    """
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Moving MNIST 数据集通常可以从以下URL获取
    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    save_path = os.path.join(download_dir, "mnist_test_seq.npy")

    print(f"Downloading Moving MNIST dataset from {url}...")

    # 使用流式下载并显示进度条
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print(f"Dataset downloaded to {save_path}")

def load_moving_mnist_data(file_path: str) -> np.ndarray:
    """
    Loads the Moving MNIST data from the .npy file.

    :param file_path: Path to the .npy file.
    :return: Numpy array containing the video sequences.
    """
    data = np.load(file_path)
    # 数据格式通常是 (num_frames, batch_size, 64, 64)
    # 我们需要转置为 (batch_size, num_frames, 64, 64)
    data = np.transpose(data, (1, 0, 2, 3))
    return data

def create_moving_mnist_training_data(data: np.ndarray, output_dir: str, num_samples: int = 1000) -> None:
    """
    Creates training data (GIFs and text files) from Moving MNIST data.

    :param data: Numpy array containing the video sequences.
    :param output_dir: Directory to save the training data.
    :param num_samples: Number of samples to process.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Creating Moving MNIST training data...")

    # 限制处理的数量
    num_samples = min(num_samples, data.shape[0])

    for i in tqdm(range(num_samples), desc="Processing", ncols=100):
        # 获取一个视频序列 (num_frames, 64, 64)
        video = data[i]

        # 创建GIF路径和文本路径
        gif_path = os.path.join(output_dir, f"mnist_{i}.gif")
        txt_path = os.path.join(output_dir, f"mnist_{i}.txt")

        # 将numpy数组转换为GIF
        # 需要将单通道数据转换为3通道(RGB)
        video_rgb = np.repeat(video[:, :, :, np.newaxis], 3, axis=3)

        # 使用moviepy创建GIF
        clip = ImageSequenceClip(list(video_rgb), fps=10)
        clip.write_gif(gif_path, program='ffmpeg', verbose=False)

        # 创建简单的文本描述
        with open(txt_path, 'w') as f:
            f.write("Moving MNIST digits")

def main():
    # 设置下载目录
    download_dir = './moving_mnist_data'
    output_dir = '../training_data'

    # 1. 下载数据集
    download_moving_mnist(download_dir)

    # 2. 加载数据
    data_path = os.path.join(download_dir, "mnist_test_seq.npy")
    data = load_moving_mnist_data(data_path)

    # 3. 创建训练数据
    create_moving_mnist_training_data(data, output_dir, num_samples=1000)

if __name__ == "__main__":
    main()