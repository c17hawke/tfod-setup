# Steps to configure tensorflow object detection-

## Download the following- 

1. [Object detection tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection)   

2. [Download](https://github.com/tensorflow/models/tree/v1.13.0) v1.13.0 model

3. [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)  faster_rcnn_inception_v2_coco or any other model of your choice

4. [Download](https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing) Dataset & utils


5. [Download](https://tzutalin.github.io/labelImg/) labelImg tool

extract all the above zip files into a tfod folder

## Creating virtual env using conda

!!! note "Commands"

    for specific python version

    ```
    conda create -n your_env_name python=3.6
    ```
    for latest python version
    ```
    conda activate your_env_name
    ```

## install the following packages - 

### for GPU
```
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow-gpu==1.14.0
```
### for non GPU 
```
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
```

---

For protobuff to py conversion download from [here](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-win64.zip) for windows

```
cd C:\tensorflow1\models\research
```

## linux mac
```
protoc object_detection/protos/*.proto --python_out=.
```

## windows
```
protoc object_detection/protos/*.proto --python_out=.
```

## conda package manager
```
conda install -c anaconda protobuf
```

## Paste all things present in utils into research folder

## Paste model foler into research folder
Now cd to research folder and run the following python file -
```
python xml_to_csv.py
```

## run the following to generate train and test records
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
```

```
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

## copy from research/object_detection/samples/config/ faster_rcnn_inception_v2_coco.config into research/training

## changes to be maid in config file are highlighted below-

### update no. of classes-
``` hl_lines="3"
model {
  ssd {
    num_classes: 6
    box_coder {
      faster_rcnn_box_coder {
```
### update no. of steps-
``` hl_lines="1"
  num_steps: 20
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
}
```
### update input path and label map path
``` hl_lines="3 5 17 19"
train_input_reader: {
  tf_record_input_reader {
    input_path: "train.record"
  }
  label_map_path: "training/labelmap.pbtxt"
}

eval_config: {
  num_examples: 8000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "test.record"
  }
  label_map_path: "training/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
}
```

## from research/object_detection/lecgacy/ copy train.py to research folder

## copy deplyment and nets folder from research/slim into research

## NOW Run from research folder. This will start ur training local system-

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
```
