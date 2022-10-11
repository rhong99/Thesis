# extract faces from frames


import cv2
import os


def extract(load_path, save_path, name, resolution=256):
    # adding more of the frame around the face
    c = 25

    image = cv2.imread(load_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    counter = 0

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = image[y-c:y+h+c, x-c:x+w+c]
        if roi_color.size != 0:
            resize_image = cv2.resize(roi_color, (resolution, resolution))
            cv2.imwrite(save_path + name + '_' + str(counter) + '.jpg', resize_image)
            counter += 1


def main():
    emotions = ['anger',
                'anxiety',
                'contempt',
                'disgust',
                'fear',
                'happy',
                'neutral',
                'sad',
                'surprise']

    print(os.path.abspath(os.getcwd()))

    for emotion in emotions:
        print('Emotions: {}'.format(emotion))
        print('')
        frames_path = '/frames/' + emotion + '/'
        save_path = os.path.abspath(os.getcwd()) + '/aligned_frames/' + emotion + '/'

        for file in os.listdir(os.path.abspath(os.getcwd()) + frames_path):
            if file.endswith('.jpg'):
                file_name = file.split('.')[0]
                extract(os.path.abspath(os.getcwd()) + '/frames/' + emotion + '/' + file,
                        save_path,
                        file_name)


main()
