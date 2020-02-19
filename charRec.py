from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


img_rows, img_cols = 32,32

train_dir = "./simpsons/train"
test_dir = "./simpsons/validation"

train_datagen = ImageDataGenerator(
    rescale= 1. /255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip= True,
    rotation_range=30,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1. /255)

batch_size = 16
num_classes = 20

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (img_rows, img_cols),
    batch_size= batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size= batch_size,
    class_mode="categorical"
)

model = Sequential()

model.add(Conv2D(64, (3,3), padding="same", input_shape=(img_rows, img_cols,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding="same", input_shape=(img_rows, img_cols,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding="same",))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

# print(model.summary())

checkpoint = ModelCheckpoint(
    "character.h5",
    save_best_only= True,
    mode= "min",
    monitor="val_loss",
    verbose=1
)

early = EarlyStopping(
    min_delta=0,
    patience=3, 
    verbose=1,
    restore_best_weights= True
)

reducelr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1, 
    min_delta=0.00001

)

callbacks = [checkpoint, reducelr, early]

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.01), metrics=["accuracy"])

nb_train=19548
nb_test = 990
epochs = 10

model.fit(
    train_generator,
    steps_per_epoch= nb_train // batch_size,
    epochs=epochs,
    callbacks= callbacks,
    validation_data=test_generator,
    validation_steps= nb_test // batch_size
)