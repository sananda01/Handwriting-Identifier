import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

#############
##CNN MODEL##
#############
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#starting the CNN
train_images, test_images = train_images/255.0, test_images/255.0 #normalizing the data to make sure that all the pixel values are not between 0 and 1
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#compling and training the model
model.compile (optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, train_labels, epochs=2, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


#################
##LETTER FINDER##
#################

image=cv2.imread(r"C:\Users\sanan_hkpez7v\OneDrive\Pictures\handwriting.png", cv2.IMREAD_GRAYSCALE)

plt.subplot(1, 2, 1)
plt.title("Sharpened")
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) #sharpening the image to reduce any noise using a kernel
sharpened_img = cv2.filter2D(image, -1, kernel)
img_f = cv2.medianBlur(sharpened_img, 5)

_, final_img = cv2.threshold( #thresh is done to get rid of grey tones so that each pixel is either black or white
    img_f,
    127,
    255,
    cv2.THRESH_BINARY_INV #BINRY_INV is used so bg is black and writing is white - opposite of og image but it works better for opencv
)

print(type(final_img))
print(final_img.shape)
print(final_img.dtype)

contours, _ = cv2.findContours(
    final_img,
    cv2.RETR_EXTERNAL, #if it is a letter with a hole in it, like o, then it will only give the outside outline, not the hole in the middle as well
    cv2.CHAIN_APPROX_SIMPLE
)
final_image_colour = cv2.cvtColor(
    final_img,
    cv2.COLOR_GRAY2BGR #changes back to colour as the rectangles are drawn in colour
)


for contour in contours:
    word=""
    x, y, w, h = cv2.boundingRect(contour)
    if w < 5 or h < 5:
        continue

    cv2.rectangle(
        final_image_colour,
        (x, y), #top left corner
        (x+w, y+h), #bottom right corner
        (0,255,0), #green rectangle as RGB
        2 #line thickness
        )
    letter = final_img[y:y+h, x:x+w] #cropping the letter from the image using the coordinates of the rectangle
    letter = cv2.resize(letter, (28,28)) #resizing the letter to the same size as the training data
    letter= letter/255.0 #normalizing the letter to make sure that all the pixel values are not between 0 and 1
    letter = letter.reshape(28,28,1) #reshaping the letter to the same shape as the training data
    answer=model.predict(letter) #predicting the letter in the rectangle by reshaping it to the same shape as the training data and then using the model to predict it
    word= word+str(np.argmax(answer))

plt.figure(figsize=(10,5))
plt.imshow(
    cv2.cvtColor(
        final_image_colour,
        cv2.COLOR_BGR2RGB
    )
)
plt.show()

print("Contours found:", len(contours))

