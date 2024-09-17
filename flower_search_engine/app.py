from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import shutil

import numpy as np
import os
import time

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

# Load the model
model_path = 'search_image_engine_model.keras'
model = load_model(model_path)

# Folder paths
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# train dir
train_dir = 'dataset/train'

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword'].lower()
    # Tạo danh sách ảnh ứng với phân lớp trùng với từ khóa
    images = []
    class_dir = os.path.join(train_dir, keyword).replace('\\', '/')
    absolute_path = os.path.abspath(class_dir)
    # Kiểm tra nếu thư mục phân lớp tồn tại
    if os.path.exists(absolute_path):
        # Lấy tất cả các ảnh trong thư mục phân lớp đó và thay thế gạch ngược bằng gạch chéo
        images = [os.path.join('dataset/train', keyword, img).replace('\\', '/') for img in os.listdir(absolute_path) if img.endswith(('.jpg', '.png'))]
    else:
        # Nếu không tìm thấy phân lớp
        images = []
    return jsonify({'images': images})

# @app.route('/upload', methods=['POST'])
# def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        # Đảm bảo thư mục uploads tồn tại
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Lưu ảnh vào thư mục uploads
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        file.save(file_path)
        
        # Dự đoán phân lớp của ảnh
        image = process_image(file_path)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Lấy phân lớp dự đoán
        
        # Map lại phân lớp dự đoán với tên các loài hoa
        class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Ví dụ các nhãn lớp
        predicted_label = class_labels[predicted_class]

        # Thêm ảnh vào tập dữ liệu tương ứng với lớp dự đoán
        destination_dir = os.path.join('dataset/train', predicted_label).replace('\\', '/') 
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        # Di chuyển file đến thư mục lớp phù hợp
        new_file_path = os.path.join(destination_dir, filename).replace('\\', '/') 

        # Kiểm tra nếu tệp đã tồn tại, nếu có thì tạo tên mới
        if os.path.exists(new_file_path):
            base_name, ext = os.path.splitext(filename)
            # Thêm timestamp để tạo tên tệp mới duy nhất
            new_file_path = os.path.join(destination_dir, f"{base_name}_{int(time.time())}{ext}").replace('\\', '/') 
        
        shutil.move(file_path, new_file_path)

        # Cập nhật mô hình bằng học tăng cường
        train_new_image(new_file_path, predicted_label)

        return f'File uploaded and classified as {predicted_label}', 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        # Đảm bảo thư mục uploads tồn tại
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Lưu ảnh vào thư mục uploads
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        file.save(file_path)
        
        # Dự đoán phân lớp của ảnh
        image = process_image(file_path)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Lấy phân lớp dự đoán
        
        # Map lại phân lớp dự đoán với tên các loài hoa
        class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Ví dụ các nhãn lớp
        predicted_label = class_labels[predicted_class]

        # Thêm ảnh vào tập dữ liệu tương ứng với lớp dự đoán
        destination_dir = os.path.join('dataset/train', predicted_label).replace('\\', '/') 
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        # Di chuyển file đến thư mục lớp phù hợp
        new_file_path = os.path.join(destination_dir, filename).replace('\\', '/') 

        # Kiểm tra nếu tệp đã tồn tại, nếu có thì tạo tên mới
        if os.path.exists(new_file_path):
            base_name, ext = os.path.splitext(filename)
            # Thêm timestamp để tạo tên tệp mới duy nhất
            new_file_path = os.path.join(destination_dir, f"{base_name}_{int(time.time())}{ext}").replace('\\', '/') 
        
        shutil.move(file_path, new_file_path)

        # Cập nhật mô hình bằng học tăng cường
        train_new_image(new_file_path, predicted_label)

        # Thực hiện tìm kiếm trong thư mục tương ứng với phân lớp đã dự đoán
        images = []
        class_dir = os.path.join(train_dir, predicted_label).replace('\\', '/')
        absolute_path = os.path.abspath(class_dir)
        
        if os.path.exists(absolute_path):
            images = [os.path.join('dataset/train', predicted_label, img).replace('\\', '/') 
                      for img in os.listdir(absolute_path) if img.endswith(('.jpg', '.png'))]

        # Trả về kết quả tìm kiếm dưới dạng JSON
        return jsonify({
            'message': f'Prediction: {predicted_label}',
            'uploaded_image': os.path.join('dataset/train', predicted_label, filename).replace('\\', '/'),
            'images': images
        }), 200


def train_new_image(image_path, label):
    """
    Hàm tái huấn luyện mô hình với ảnh mới.
    """
    # Định nghĩa đường dẫn tới thư mục huấn luyện
    train_dir = 'dataset/train'

    # Tạo ImageDataGenerator để lấy thông tin nhãn
    datagen = ImageDataGenerator(rescale=1./255)

    # Tạo train_generator để lấy thông tin về nhãn
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Load ảnh mới vừa thêm vào
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    img_array = img_array / 255.0  # Chuẩn hóa ảnh

    # Tạo nhãn dạng one-hot encoding từ tên phân lớp (ví dụ như 'daisy', 'sunflower')
    if label not in train_generator.class_indices:
        raise ValueError(f"Label '{label}' không tồn tại trong tập dữ liệu huấn luyện.")
    
    label_index = train_generator.class_indices[label]
    label_one_hot = np.zeros((1, len(train_generator.class_indices)))
    label_one_hot[0, label_index] = 1

    # Huấn luyện lại mô hình chỉ với ảnh mới
    model.fit(img_array, label_one_hot, epochs=1)

    # Lưu mô hình đã cập nhật
    model.save('models/search_image_engine_model.keras')

    print(f"Đã huấn luyện mô hình với ảnh {image_path} và nhãn {label}.")


if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')
