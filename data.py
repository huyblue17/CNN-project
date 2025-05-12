import numpy as np
import tensorflow as tf

def load_mnist_data():
    """
    Tải dữ liệu MNIST từ tensorflow.
    Returns: (train_images, train_labels), (test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels):
    """
    Chuẩn hóa và reshape dữ liệu.
    Args:
        images: numpy array, shape (N, 28, 28)
        labels: numpy array, shape (N,)
    Returns: (processed_images, processed_labels)
    """
    images = images.astype('float32') / 255.0
    images = images.reshape(-1, 28, 28, 1)
    labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    return images, labels

if __name__ == "__main__":
    # Kiểm tra dữ liệu
    raw_train_images, raw_train_labels, raw_test_images, raw_test_labels = load_mnist_data()
    print("Kích thước tập train thô:", raw_train_images.shape, raw_train_labels.shape)
    print("Kích thước tập test thô:", raw_test_images.shape, raw_test_labels.shape)
    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)
    test_images, test_labels = preprocess_data(raw_test_images, raw_test_labels)
    print("Kích thước tập train đã xử lý:", train_images.shape, train_labels.shape)
    print("Kích thước tập test đã xử lý:", test_images.shape, test_labels.shape)
