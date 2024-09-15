import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Đường dẫn dữ liệu
train_dir = 'flower_search_engine/static/dataset/flowers/train'
test_dir = 'flower_search_engine/static/dataset/flowers/test'
model_save_path = 'flower_search_engine/models/search_image_engine_model.keras'

# Kiểm tra sự tồn tại của thư mục dữ liệu
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Thư mục huấn luyện không tồn tại: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Thư mục kiểm tra không tồn tại: {test_dir}")

# Kích thước hình ảnh
img_width, img_height = 224, 224

# Tạo đối tượng ImageDataGenerator để tăng cường dữ liệu và tiền xử lý
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Tạo các bộ dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'  # Chuyển đổi nhãn thành One-Hot Encoding
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'  # Chuyển đổi nhãn thành One-Hot Encoding
)

# Tải mô hình VGG16 đã được huấn luyện trên ImageNet, không bao gồm lớp đầu ra
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Đóng băng các lớp của mô hình cơ sở
for layer in base_model.layers:
    layer.trainable = False

# Thêm các lớp tùy chỉnh phía trên mô hình VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Tạo mô hình cuối cùng
model = Model(inputs=base_model.input, outputs=predictions)

# Biên dịch mô hình với categorical_crossentropy cho bài toán phân loại nhiều lớp
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Sử dụng categorical_crossentropy
    metrics=['accuracy']
)

# Tạo checkpoint để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint]
)

print("Huấn luyện hoàn tất và mô hình đã được lưu tại", model_save_path)
