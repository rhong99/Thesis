# extract frames from video


import cv2
import os


# frequency is the frame interval (e.g. 6 == extract 1 frame every 6)
def extract(path, name, save_path, frequency=6):
    video = cv2.VideoCapture(path)

    frame_num = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if (frame_num % frequency) == 0:
            cv2.imwrite(save_path + name + str(frame_num) + '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        frame_num += 1

    video.release()
    cv2.destroyAllWindows()


def main():
    # emotion classes
    emotions = ['anger',
                'anxiety',
                'contempt',
                'disgust',
                'fear',
                'happy',
                'neutral',
                'sad',
                'surprise']

    for emotion in emotions:
        print('Emotions: {}'.format(emotion))
        print('')
        emotion_path = '/video/' + emotion + '/'
        save_path = os.path.abspath(os.getcwd()) + '/frames/' + emotion + '/'

        for file in os.listdir(os.path.abspath(os.getcwd()) + emotion_path):
            if file.endswith('.mp4'):
                print(file)
                print('Processing: {}'.format(file))
                print('')
                file_name = file.split('.')[0]
                video_path = os.path.abspath(os.getcwd()) + emotion_path + file
                extract(video_path, file_name, save_path)

        print('')
        print('------------')
        print('')

    print('Complete')


main()
