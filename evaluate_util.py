import numpy as np
import time

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-faster_rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-faster_rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-faster_rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def all_annotation_from_instance(instance):
    all_annotation = [[], []]
    for i in range(len(instance['ann']['labels'])):
        all_annotation[instance['ann']['labels'][i]].append(instance['ann']['bboxes'][i])
    return [np.array(all_annotation[0]), np.array(all_annotation[1])]


def evaluate(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    # all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]  # [[bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)], [bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)]]
    # all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            detections = all_detections[i][label]
            annotations = np.array(all_annotations[i][label])
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
        total_instances.append(num_annotations)

    return average_precisions, total_instances


def get_false_positive(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    false_positives = []
    for i in range(len(all_annotations)):
        print(i, end='\r')
        detections = all_detections[i]
        annotations = np.array(all_annotations[i])
        detected_annotations = []

        for d in detections:

            if annotations.shape[0] == 0:
                continue

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                pass
            else:
                false_positives.append(d)
    return false_positives
