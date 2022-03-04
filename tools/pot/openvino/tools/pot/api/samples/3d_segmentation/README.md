# API usage sample for segmentation task {#pot_sample_3d_segmentation_README}

This sample demonstrates the use of the [Post-training Optimization Tool API](@ref pot_compression_api_README) for the task of quantizing a 3D segmentation model.
The [Brain Tumor Segmentation](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.md) model from PyTorch* is used for this purpose.
A custom `DataLoader` is created to load images in NIfTI format from [Medical Segmentation Decathlon BRATS 2017](http://medicaldecathlon.com/) dataset for 3D semantic segmentation task 
and the implementation of Dice Index metric is used for the model evaluation. In addition, this sample demonstrates how one can use image metadata obtained during image reading and 
preprocessing to post-process the model raw output.

## How to prepare the data

To run this sample, you will need to download the Brain Tumors 2017 part of the Medical Segmentation Decathlon image database http://medicaldecathlon.com/.
3D MRI data in NIfTI format can be found in the `imagesTr` folder, and segmentation masks are in `labelsTr`.


## How to Run the Sample
In the instructions below, the Post-Training Optimization Tool directory `<POT_DIR>` is referred to:
- `<ENV>/lib/python<version>/site-packages/` in the case of PyPI installation, where `<ENV>` is a Python* 
  environment where OpenVINO is installed and `<version>` is a Python* version, e.g. `3.6`.
- `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` in the case of OpenVINO distribution package. 
  `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. To get started, follow the [Installation Guide](@ref pot_InstallationGuide).
2. Launch [Model Downloader](@ref omz_tools_downloader) tool to download `brain-tumor-segmentation-0002` model from the Open Model Zoo repository.
   ```sh
   python3 ./downloader.py --name brain-tumor-segmentation-0002
   ```
3. Launch [Model Converter](@ref omz_tools_downloader) tool to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 ./converter.py --name brain-tumor-segmentation-0002 --mo <PATH_TO_MODEL_OPTIMIZER>/mo.py
   ```
4. Launch the sample script:
   ```sh
   python3 <POT_DIR>/api/samples/3d_segmentation/3d_segmentation_sample.py -m <PATH_TO_IR_XML> -d <BraTS_2017/imagesTr> --mask-dir <BraTS_2017/labelsTr>
   ```
   Optional: you can specify .bin file of IR directly using the `-w`, `--weights` options.
