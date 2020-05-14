# Steps to configure tensorflow object detection-

## STEP-1 Download the following content- 

1. [Download](https://github.com/tensorflow/models/tree/v1.13.0) v1.13.0 model

2. [Download](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) ssd_mobilenet_v1_coco model from model zoo **or** any other model of your choice from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md" target="_blank">tensorflow model zoo.</a>

3. [Download](https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing) Dataset & utils


4. [Download](https://tzutalin.github.io/labelImg/) labelImg tool for labeling images.

before extraction your should have a the following compressed files - 

![zipFiles](img/zipFiles.png)

---

## STEP-2 extract all the above zip files into a tfod folder and remove the compressed files. 

Now you should have the following folders -

![firstExtaction](img/firstExtaction.png)

---


## STEP-3 Creating virtual env using conda

!!! note "Commands"

    for specific python version

    ```
    conda create -n your_env_name python=3.6
    ```
    for latest python version
    ```
    conda activate your_env_name
    ```

---


## STEP-4 Install the following packages in your new environment- 

### for GPU
```
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow-gpu==1.14.0
```
### for CPU only 
```
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
```

---


## STEP-5 For protobuff to .py conversion download from tool from here -

For windows -> [download](https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-win64.zip)
source for other versions and OS - <a href="https://github.com/protocolbuffers/protobuf/releases/tag/v3.11.4" target="_blank">click here</a> 

Open a commmand line and cd to research folder

Now in research folder run the following command-

### For Linux or Mac
```
protoc object_detection/protos/*.proto --python_out=.
```

### For Windows
```
protoc object_detection/protos/*.proto --python_out=.
```

---


## STEP-6 install protobuf using conda package manager
```
conda install -c anaconda protobuf
```

---


## STEP-7 Paste all content present in utils into research folder

Following are the files and folder present in the utils folder-

![utils folder structure](img/underUtilsFolder.png)


---


## STEP-8 Paste SSD or fasterRCNN folder or any other model downloaded from model zoo into research folder-

Now cd to research folder and run the following python file -

```
python xml_to_csv.py
```

---


## STEP-9 Run the following to generate train and test records
from research folder-
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
```

```
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

---


## STEP-10 Copy from _research/object_detection/samples/config/_ _YOURMODEL.config_ file into _research/training_


### Changes to be made in config file are highlighted below-

!!! Note
    Following config file shown here is with respect to ssd_mobilenet_v1_coco.So if you have downloaded it for any other model apart from SSD you'll see config file as shown below-
    ```
    model {
    YOUR_MODEL_NAME {
      num_classes: 6
      box_coder {
        faster_rcnn_box_coder {
    ```

#### Update no. of classes-
``` hl_lines="3"
model {
  ssd {
    num_classes: 6
    box_coder {
      faster_rcnn_box_coder {
```
#### Update no. of steps-
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
#### Update input path and label map path
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

---


## STEP-11 From _research/object_detection/lecgacy/_ copy _train.py_ to research folder

legacy folder contains train.py as shown below - 
![legacy folder](img/legacyFolder.png)

---


## STEP-12 Copy _deployment_ and _nets_ folder from _research/slim_ into research folder

slim folder contains the following folders -

![slim folder](img/slimFolder.png)


---

## STEP-13 NOW Run from research folder. This will start the training in your local system-

!!! NOTE
    copy the command and replace **YOUR_MODEL.config** with your own models name for example **ssd_mobilenet_v1_coco**
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/YOUR_MODEL.config
```

---

!!! Warning
    Always run all the commands in research folder