import os
import tensorflow as tf
from data import load_mnist_data, preprocess_data

def build_cnn_model():
    """
    Tạo mô hình CNN cho nhận diện chữ số MNIST.
    Returns: Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels):
    """
    Huấn luyện mô hình.
    Args:
        model: Keras model
        train_images: numpy array, shape (N, 28, 28, 1)
        train_labels: numpy array, shape (N, 10)
    Returns: Training history
    """
    history = model.fit(train_images, train_labels,
                       epochs=5,
                       batch_size=64,
                       validation_split=0.1,
                       verbose=1)
    return history

if __name__ == "__main__":
    print("Đang tải dữ liệu MNIST...")
    raw_train_images, raw_train_labels, _, _ = load_mnist_data()

    print("Đang tiền xử lý dữ liệu...")
    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)

    print("Đang xây dựng mô hình CNN...")
    model = build_cnn_model()
    model.summary()

    print("Đang huấn luyện mô hình...")
    history = train_model(model, train_images, train_labels)

    # Lưu mô hình trực tiếp vào thư mục hiện tại
    model_path = "mnist_cnn_model.h5"
    print(f"Đang lưu mô hình vào '{model_path}'...")
    model.save(model_path)
    print(f"Mô hình đã được lưu vào '{model_path}'")
