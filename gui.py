import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps

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

def create_digit_recognition_gui(model):
    # Tạo cửa sổ chính
    window = tk.Tk()
    window.title("Nhận diện chữ số viết tay")

    # --- UI/UX Enhancements ---
    # Title label
    title_label = tk.Label(window, text="Nhận diện chữ số viết tay MNIST", font=('Arial', 18, 'bold'), fg='#2c3e50')
    title_label.pack(pady=(15, 5))

    # Instruction label
    instruction_label = tk.Label(window, text="Vẽ một chữ số (0-9) vào khung bên dưới, sau đó nhấn 'Dự đoán' để nhận kết quả.", font=('Arial', 11), fg='#34495e')
    instruction_label.pack(pady=(0, 10))

    # Tạo canvas để vẽ
    canvas_width = 280
    canvas_height = 280
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white", highlightthickness=2, highlightbackground="#bdc3c7")
    canvas.pack(pady=10)

    # Biến lưu nét vẽ và lịch sử dự đoán
    drawing_strokes = []
    history_log = []

    # --- Button Frame for better layout ---
    button_frame = tk.Frame(window)
    button_frame.pack(pady=5)
    predict_button = ttk.Button(button_frame, text="Dự đoán", command=lambda: recognize_digit())
    predict_button.grid(row=0, column=0, padx=8)
    clear_button = ttk.Button(button_frame, text="Xóa", command=lambda: clear_canvas())
    clear_button.grid(row=0, column=1, padx=8)
    save_button = ttk.Button(button_frame, text="Lưu lịch sử", command=lambda: save_history())
    save_button.grid(row=0, column=2, padx=8)

    # Tạo nhãn hiển thị kết quả
    result_label = tk.Label(window, text="Kết quả: Chưa dự đoán", font=('Arial', 14), fg='#2980b9', justify='left', anchor='w')
    result_label.pack(pady=10, fill='x', padx=20)

    # --- Status Bar ---
    status_var = tk.StringVar(value="Sẵn sàng.")
    status_bar = tk.Label(window, textvariable=status_var, bd=1, relief='sunken', anchor='w', font=('Arial', 10), fg='#7f8c8d')
    status_bar.pack(side='bottom', fill='x')

    def start_drawing(event):
        """
        Bắt đầu vẽ khi nhấn chuột trái.
        Args:
            event: Sự kiện chuột (tọa độ x, y)
        """
        drawing_strokes.append([(event.x, event.y)])
        canvas.old_coords = (event.x, event.y)

    def draw_line(event):
        """
        Vẽ nét liên tục khi kéo chuột.
        Args:
            event: Sự kiện chuột (tọa độ x, y)
        """
        x, y = event.x, event.y
        x1, y1 = canvas.old_coords
        canvas.create_line(x1, y1, x, y, width=15, fill="black", capstyle="round", smooth=True)
        drawing_strokes[-1].append((x, y))
        canvas.old_coords = (x, y)

    def clear_canvas():
        """
        Xóa canvas và reset kết quả.
        """
        canvas.delete("all")
        drawing_strokes.clear()
        result_label.config(text="Kết quả: Chưa dự đoán")
        status_var.set("Đã xóa bảng vẽ.")

    def recognize_digit():
        """
        Chuyển nét vẽ thành ảnh, dự đoán chữ số, hiển thị kết quả.
        """
        if not drawing_strokes:
            result_label.config(text="Vui lòng vẽ một chữ số trước!")
            status_var.set("Chưa có nét vẽ để dự đoán.")
            return
        # Tạo ảnh từ nét vẽ
        image_pil = Image.new("L", (canvas_width, canvas_height), color=255)
        draw = ImageDraw.Draw(image_pil)
        for stroke in drawing_strokes:
            if len(stroke) > 1:
                draw.line(stroke, fill=0, width=15)
        image_pil = ImageOps.invert(image_pil)

        # Lưu ảnh vào outputs/
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        image_path = os.path.join(outputs_dir, "drawn_digit.png")
        image_pil.save(image_path)

        # Chuẩn bị ảnh cho mô hình
        image_pil = image_pil.resize((28, 28), Image.Resampling.LANCZOS)
        image_np = np.array(image_pil) / 255.0
        image_np = image_np.reshape(1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(image_np, verbose=0)
        predicted_label = np.argmax(prediction)
        history_log.append(predicted_label)
        print("Lịch sử dự đoán:", history_log)

        # Hiển thị kết quả
        probs = prediction[0]
        prob_text = "\n".join([f"Class {i}: {probs[i]:.4f}" for i in range(10)])
        result_label.config(text=f"Kết quả: {predicted_label}\n\nXác suất:\n{prob_text}")
        status_var.set(f"Dự đoán: {predicted_label}")

        # Trực quan hóa ảnh vẽ
        plt.figure(figsize=(3, 3))
        plt.imshow(image_np.reshape(28, 28), cmap='gray')
        plt.title(f"Dự đoán: {predicted_label}")
        plt.axis('off')
        plt.show()

    def save_history():
        """
        Lưu lịch sử dự đoán vào file.
        """
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        history_path = os.path.join(outputs_dir, "prediction_history.txt")
        with open(history_path, 'w') as f:
            f.write(str(history_log))
        print(f"Lịch sử dự đoán đã được lưu vào '{history_path}'")
        status_var.set("Lịch sử dự đoán đã được lưu.")

    # Gán sự kiện cho canvas
    canvas.bind('<Button-1>', start_drawing)
    canvas.bind('<B1-Motion>', draw_line)

    # Chạy giao diện
    window.mainloop()

if __name__ == "__main__":
    print("Đang tìm mô hình...")
    model_path = find_h5_file(os.getcwd())
    if model_path:
        print(f"Mô hình được tìm thấy tại: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Không tìm thấy mô hình 'mnist_cnn_model.h5'. Vui lòng chạy train.py trước!")
        exit(1)

    print("Đang khởi động giao diện Tkinter...")
    create_digit_recognition_gui(model)
