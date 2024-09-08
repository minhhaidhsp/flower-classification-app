from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tải mô hình ResNet50 với trọng số đã được huấn luyện trên ImageNet
base_model = ResNet50(weights='imagenet', include_top=False)

# Thêm các lớp mới cho bài toán phân loại loài hoa
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # 10 lớp tương ứng với 10 loại hoa

# Tạo mô hình mới
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng tất cả các lớp của ResNet50 để không huấn luyện lại từ đầu
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tạo ImageDataGenerator cho huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('dataset/flowers/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Huấn luyện mô hình (fine-tuning)
model.fit(train_generator, epochs=100, steps_per_epoch=100)


# Lưu toàn bộ mô hình vào một file HDF5
model.save('flower_classification_model.h5')