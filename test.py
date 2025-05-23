import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import load_mnist_data, preprocess_data

def find_h5_file(root_dir):
    """
    Dò tìm file mnist_cnn_model.h5 trong thư mục root_dir và các subfolder.
    Args:
        root_dir: Thư mục gốc để tìm
    Returns: Đường dẫn đến file .h5 hoặc None nếu không tìm thấy
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "mnist_cnn_model.h5":
                return os.path.join(dirpath, filename)
    return None

def evaluate_on_test(model, raw_test_images, raw_test_labels, processed_test_images, processed_test_labels):
    print("\nRating on raw dataset:")
    temp_images, temp_labels = preprocess_data(raw_test_images, raw_test_labels)
    raw_loss, raw_accuracy = model.evaluate(temp_images, temp_labels, verbose=0)
    print(f"Accuracy on raw dataset: {raw_accuracy:.4f}")

    print("\nRating on processed test dataset:")
    processed_loss, processed_accuracy = model.evaluate(processed_test_images, processed_test_labels, verbose=0)
    print(f"Accuracy on processed test dataset: {processed_accuracy:.4f}")

def visualize_predictions(model, test_images, test_labels):
    num_samples = 50
    indices = np.random.choice(test_images.shape[0], size=num_samples, replace=False)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    predictions = model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)

    plt.figure(figsize=(20, 20))
    for i in range(num_samples):
        plt.subplot(10, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='binary')
        plt.title(f"Predict: {predicted_classes[i]}\nActual: {true_classes[i]}",
                  color='green' if predicted_classes[i] == true_classes[i] else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading data MNIST...")
    _, _, raw_test_images, raw_test_labels = load_mnist_data()

    print("Process data...")
    test_images, test_labels = preprocess_data(raw_test_images, raw_test_labels)

    print("Finding model...")
    model_path = find_h5_file(os.getcwd())
    if model_path:
        print(f"Mode at: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Can't find model 'mnist_cnn_model.h5'. Compile train.py first!")
        exit(1)

    print("Rating model on test dataset...")
    evaluate_on_test(model, raw_test_images, raw_test_labels, test_images, test_labels)

    print("Result of prediction...")
    visualize_predictions(model, test_images, test_labels)
