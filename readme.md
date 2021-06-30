# Download the weights here
https://drive.google.com/drive/folders/1fVpx032B1U9jzCvKc_I0TU2peBf9o02C?usp=sharing

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
   * The `config` is for the corresponding config files and weights, and also.
   the output folder's name. 
   You can refer to both `mmpredict` and `mmeval` for which weight 
   should be used with which config. The output folder's name should not be duplicated.
   
 * `mmpredict_field.py` is simpler version of `mmpredict`. One weight one config.
 Predict from a folder and output to a folder.
# Evaluation
 * The `mmeval.py` is for predicting all images in `test.json`(download link above)
 and preform mAP evaluation with IOU 0.3 - 0.9. It prints the results and output
 to a json file.
   * Note that there's a comment at line 117. Uncomment that will save all prediction
   in the json alnog with the results.
   