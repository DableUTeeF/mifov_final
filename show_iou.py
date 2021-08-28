import json
import cv2
from boxutils import add_bbox
import numpy as np
from evaluate_util import compute_overlap
if __name__ == '__main__':
    dataset = json.load(open('/home/palm/PycharmProjects/mmdetection/anns/test.json'))
    for data in dataset:
        s = 3
        image = cv2.imread(data['filename'])
        image = cv2.resize(image, None, None, s, s)
        bbox = np.array(data['ann']['bboxes'])

        cv2.imshow('a', image[605*s:670*s, 635*s:730*s])

        # true
        add_bbox(image, bbox[0]*s, 'true', (0, 180, 0), show_txt=False)
        print('a')

        # predict
        # add_bbox(image, [658*s, 626*s, 706*s, 652*s], '0.9', (180, 0, 0), pos='bot', show_txt=False)
        # add_bbox(image, [658*s, 626*s, 700*s, 649*s], '0.7', (100, 255, 255), pos='bot', show_txt=False)
        add_bbox(image, [653*s, 619*s, 704*s, 647*s], '0.5', (100, 100, 255), pos='bot', show_txt=False)
        # add_bbox(image, [658*s, 626*s, 688*s, 640*s], '0.3', (255, 50, 180), pos='bot', show_txt=False)

        cv2.imshow('b', image[605*s:670*s, 635*s:730*s])
        cv2.waitKey()
