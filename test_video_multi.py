import numpy as np
import cv2
import time
from manager import InputManager

"""
{
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted-plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv/monitor',
    255: 'ambigious'
}
"""

def read_image(img):

    w, h, _ = img.shape

    ratio = 512. / np.max([w,h])
    image_np = cv2.resize(img, (int(ratio*h),int(ratio*w)))
    resized = image_np / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized_im = np.pad(resized, ((0, pad_x),(0,0),(0,0)), mode='constant')
    return img, resized_im, pad_x


def main(src):
    vid = cv2.VideoCapture(src)
    vid.set(3, 640)
    vid.set(5, 480)

    # Instantiete model
    # 15 for filter only person, use 7 for cars 
    # This is based on the Pascal VOC dataset

    # Instantiate Model

    model = InputManager(model='seg_obj', multi=True)


    while True:
        _, frame = vid.read()
        
        t1 = time.time()

        seg_map = model.predict(frame)
        if seg_map:
            t2 = time.time()
            print('TIME that took the inference', t2 - t1)
            input_mask = seg_map.squeeze()

            cv2.imshow('frame', input_mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
        else:
            print('SEG MAP', seg_map)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    src = 0
    main(src)