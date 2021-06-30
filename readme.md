# Download the weights here
https://drive.google.com/drive/folders/1SYETX8p84YmaOszo_vr0YWfhUl02_Cn4?usp=sharing
# The dataset
 * Train: https://drive.google.com/drive/folders/1C253KHa8D6-moFILSEF-fpOSCtFmlqRv?usp=sharing
 * Val: https://drive.google.com/drive/folders/1VMxXUNnlTNnpLP3Xzku_7cc7FnfMSmhD?usp=sharing
 * The `test.json` that found in many predict files: https://drive.google.com/file/d/1CgAAVkKBrV9Cc3k1-GGUZQefrLNHF6SK/view?usp=sharing

# The predict files
 * `mmpredict.py` is for multiple models and/or weights files.
   * The `path` variable is consist of
     1. The path to [mmdetection](https://github.com/open-mmlab/mmdetection) config. The newest version is recommended as some models here
     are rather new. (I'm using 2.13.0)
     2. The path to weight folders
     3. Output path
   * The `config` is for the corresponding config files and weights, and also
   the output folder's name. 
   You can refer to both `mmpredict` and `mmeval` for which weight 
   should be used with which config. The output folder's name should not be duplicated.
   
 * `mmpredict_field.py` is simpler version of `mmpredict`. One weight one config.
 Predict from a folder and output to a folder.
 ## Note
 All these files use `filename` when calling `inference_detector()` but actually
 you can put `ndarray` in instead of string too. Like
 ```
cfg = Config.fromfile('/home/palm/PycharmProjects/pig/mmdetection/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py')
cfg.model.bbox_head.num_classes = 2
model = init_detector(cfg, '/media/palm/BiggerData/algea/weights/gfl_r50/epoch_24.pth', device='cuda')
image = cv2.imread('ov.png')
result = inference_detector(model, image)
 ```

# Training
 * Colab example https://colab.research.google.com/drive/13G_ED9aeuskYh7JbU28D1lAHYkq41DZ9?usp=sharing
 
# Evaluation
 * The `mmeval.py` is for predicting all images in `test.json`(download link above)
 and preform mAP evaluation with IOU 0.3 - 0.9. It prints the results and output
 to a json file.
   * There's a comment at line 117. Uncomment that will save all prediction
   in the json along with the results.

# The output
The `result` from `inference_detector` will be a list of `ndarray` arrange by class.
In the array are the bounding boxes in format `(x1, y1, x2, y2, score)`.

For example everything in `result[0]` will be MIF, and likewise everything in
`result[1]` will be OV. 

The boxes are in `float32` but they are the actual pixel of the input image not in 0-1.
