from flask import Flask, request, render_template
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64  # Add this import

app = Flask(__name__)

# Tạo lại train_generator để lấy thông tin về class_indices
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Tải mô hình đã huấn luyện
model = load_model('flower_classification_model.h5')
# Lấy thông tin về các lớp
class_indices = train_generator.class_indices
# Tạo dictionary ánh xạ từ chỉ số đến tên lớp
class_labels = {v: k for k, v in class_indices.items()}
# class_labels = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}


# Hàm xử lý ảnh đầu vào
def prepare_image(image):
    # Chuyển đổi ảnh thành kích thước phù hợp
    image = image.resize((224, 224)) # Thay đổi kích thước theo yêu cầu của mô hình
    # Chuyển đổi ảnh thành mảng numpy
    image = img_to_array(image)
    # Chuẩn hóa giá trị pixel
    image = image / 255.0 # Chuyển đổi ảnh thành mảng numpy và chuẩn hóa
    # Thêm một chiều để biến ảnh thành batch (batch size = 1)
    image = np.expand_dims(image, axis=0)
    
    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream)
            # Chuẩn bị ảnh cho mô hình dự đoán
            prepared_image = prepare_image(img)
            
            # Predict class
            predictions = model.predict(prepared_image)
            predicted_class = np.argmax(predictions, axis=1)
    
            # Predict Label
            predicted_label = class_labels[predicted_class[0]]

            # Convert image to base64 for display
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            return render_template('index.html', img_data=img_base64, prediction=predicted_label)
    
    return render_template('index.html', img_data=None, prediction=None)

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')