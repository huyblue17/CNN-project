import os
import json
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
    Returns: Training history
    """
    # Callbacks for better training management
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=callbacks
    )
    return history

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

if __name__ == "__main__":
    print("Đang tải dữ liệu MNIST...")
    raw_train_images, raw_train_labels, _, _ = load_mnist_data()

    print("Đang tiền xử lý dữ liệu...")
    train_images, train_labels = preprocess_data(raw_train_images, raw_train_labels)

    print("Đang xây dựng mô hình CNN...")
    model = build_cnn_model()
    model.summary()

    print("Đang huấn luyện mô hình...")
    history = train_model(model, train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

    # Lưu mô hình trực tiếp vào thư mục hiện tại
    model_path = "mnist_cnn_model.h5"
    print(f"Đang lưu mô hình vào '{model_path}'...")
    model.save(model_path)
    print(f"Mô hình đã được lưu vào '{model_path}'")

    # Lưu lịch sử huấn luyện
    save_training_history(history)
