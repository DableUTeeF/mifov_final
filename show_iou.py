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
        add_bbox(image, bbox[0]*s, 'true', (0, 180, 0), show_txt=False)  # 49, 28
        print('a')

        # predict
        add_bbox(image, [658*s, 626*s, 706*s, 652*s], '0.9', (180, 0, 0), pos='bot', show_txt=False)
        add_bbox(image, [658*s, 626*s, 700*s, 649*s], '0.7', (200, 180, 0), pos='bot', show_txt=False)
        add_bbox(image, [658*s, 626*s, 694*s, 645*s], '0.5', (100, 100, 255), pos='bot', show_txt=False)
        add_bbox(image, [658*s, 626*s, 688*s, 640*s], '0.3', (255, 50, 180), pos='bot', show_txt=False)

        # add annotation

        image = image[605*s:670*s, 635*s:730*s]
        image[140:193, 205:280, :] = 255

        image = cv2.line(image, (210, 145), (235, 145), (255, 50, 180))
        image = cv2.putText(image, 'iou=0.3', (240, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        image = cv2.line(image, (210, 155), (235, 155), (100, 100, 255))
        image = cv2.putText(image, 'iou=0.5', (240, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        image = cv2.line(image, (210, 165), (235, 165), (200, 180, 0))
        image = cv2.putText(image, 'iou=0.7', (240, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        image = cv2.line(image, (210, 175), (235, 175), (180, 0, 0))
        image = cv2.putText(image, 'iou=0.9', (240, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        image = cv2.line(image, (210, 185), (235, 185), (0, 180, 0))
        image = cv2.putText(image, 'gt', (240, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('b', image)
        cv2.waitKey()

        break
