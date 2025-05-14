import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

def train_model(model, train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1):
    """
    Huấn luyện mô hình.
    Args:
        model: Keras model
        train_images: numpy array, shape (N, 28, 28, 1)
        train_labels: numpy array, shape (N, 10)
        epochs: Số lần lặp huấn luyện
        batch_size: Kích thước batch
        validation_split: Tỷ lệ dữ liệu validation
    Returns: Training history, epoch times
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    ]

    epoch_times = []
    class TimingCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.start_time
            epoch_times.append(epoch_time)

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[callbacks, TimingCallback()]
    )
    return history, epoch_times

def save_training_history(history, output_dir="outputs"):
    """
    Lưu lịch sử huấn luyện vào file JSON.
    Args:
        history: Training history object
        output_dir: Thư mục lưu trữ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Lịch sử huấn luyện đã được lưu vào '{history_path}'")

def plot_training_history(history):
    """
    Vẽ biểu đồ so sánh độ chính xác và mất mát giữa tập train và validation.
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 5))

    # Biểu đồ 1: Độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Biểu đồ 2: Mất mát
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_validation_predictions(model, train_images, train_labels, validation_split=0.1, num_samples=10):
    """
    So sánh giá trị thực tế và dự đoán trên tập validation.
    Args:
        model: Keras model
        train_images: numpy array, shape (N, 28, 28, 1)
        train_labels: numpy array, shape (N, 10)
        validation_split: Tỷ lệ dữ liệu validation
        num_samples: Số mẫu để hiển thị
    """
    # Tính số lượng mẫu validation
    num_train = len(train_images)
    num_val = int(num_train * validation_split)
    val_images = train_images[-num_val:]
    val_labels = train_labels[-num_val:]

    # Dự đoán trên tập validation
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(val_labels, axis=1)

    # Tính độ chính xác trên tập validation
    accuracy = np.mean(predicted_classes == actual_classes)
    print(f"Độ chính xác trên tập validation: {accuracy:.4f}")

    # Hiển thị một số mẫu ví dụ
    indices = np.random.choice(len(val_images), size=num_samples, replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(val_images[idx].reshape(28, 28), cmap='binary')
        plt.title(f"Actual: {actual_classes[idx]}\nPredicted: {predicted_classes[idx]}",
                  color='green' if predicted_classes[idx] == actual_classes[idx] else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_training_speed(epoch_times):
    """
    Phân tích tốc độ huấn luyện.
    Args:
        epoch_times: List thời gian huấn luyện mỗi epoch (giây)
    """
    avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
    print(f"Thời gian huấn luyện trung bình mỗi epoch: {avg_time_per_epoch:.2f} giây")
    print("Thời gian huấn luyện mỗi epoch:")
    for i, t in enumerate(epoch_times, 1):
        print(f"Epoch {i}: {t:.2f} giây")

if __name__ == "__main__":
    print("Đang tải dữ liệu MNIST...")
    raw_train_images, raw_train_labels, _, _ = load_mnist_data()

    print("Đang tiền xử lý dữ liệu...")
    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)

    print("Đang xây dựng mô hình CNN...")
    model = build_cnn_model()
    model.summary()

    print("Đang huấn luyện mô hình...")
    history, epoch_times = train_model(model, train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

    # Lưu mô hình trực tiếp vào thư mục hiện tại
    model_path = "mnist_cnn_model.h5"
    print(f"Đang lưu mô hình vào '{model_path}'...")
    model.save(model_path)
    print(f"Mô hình đã được lưu vào '{model_path}'")

    # Lưu lịch sử huấn luyện
    save_training_history(history)

    # So sánh tập train và validation (độ chính xác và mất mát)
    print("So sánh hiệu suất giữa tập train và validation...")
    plot_training_history(history)

    # So sánh giá trị thực tế và dự đoán trên tập validation
    print("So sánh giá trị thực tế và dự đoán trên tập validation...")
    compare_validation_predictions(model, train_images, train_labels, validation_split=0.1, num_samples=10)

    # Phân tích tốc độ huấn luyện
    print("Phân tích tốc độ huấn luyện...")
    analyze_training_speed(epoch_times)
