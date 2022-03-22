# Quantizatiing Image Classification Model {#pot_example_classification_README}

This example demonstrates the use of the [Post-training Optimization Tool API](@ref pot_compression_api_README) for the task of quantizing a classification model.
The [MobilenetV2](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v2-1.0-224/mobilenet-v2-1.0-224.md) model from TensorFlow* is used for this purpose.
A custom `DataLoader` is created to load the [ImageNet](http://www.image-net.org/) classification dataset and the implementation of Accuracy at top-1 metric is used for the model evaluation.

## How to prepare the data

To run this example, you need to [download](http://www.image-net.org/download-faq) the validation part of the ImageNet image database and place it in a separate folder, 
which will be later referred as `<IMAGES_DIR>`. Annotations to images should be stored in a separate .txt file (`<IMAGENET_ANNOTATION_FILE>`) in the format `image_name label`.


## How to Run the example
In the instructions below, the Post-Training Optimization Tool directory `<POT_DIR>` is referred to:
- `<ENV>/lib/python<version>/site-packages/` in the case of PyPI installation, where `<ENV>` is a Python* 
  environment where OpenVINO is installed and `<version>` is a Python* version, for example `3.6`.
  `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. Launch [Model Downloader](@ref omz_tools_downloader) tool to download `mobilenet-v2-1.0-224` model from the Open Model Zoo repository.
   ```sh
   python3 ./downloader.py --name mobilenet-v2-1.0-224
   ```
2. Launch [Model Converter](@ref omz_tools_downloader) tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 ./converter.py --name mobilenet-v2-1.0-224 --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py
   ```
3. Launch the example script:
   ```sh
   python3 <POT_DIR>/api/examples/classification/classification_example.py -m <PATH_TO_IR_XML> -a <IMAGENET_ANNOTATION_FILE> -d <IMAGES_DIR>
   ```
   Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
