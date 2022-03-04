#  OpenVINO™ MonoDepth Python Demo

This tutorial describes the example from the following YouTube* video:
///

To learn more about how to run the MonoDepth Python* demo application, refer to the [documentation](https://docs.openvino.ai/latest/omz_demos_monodepth_demo_python.html).

Tested on OpenVINO™ 2021, Ubuntu 18.04.

## 1. Set Environment

Define the OpenVINO™ install directory:
```
export OV=/opt/intel/openvino_2022/
```
Define the working directory. Make sure the directory exist:
```
export WD=~/MonoDepth_Python/
```

## 2. Install Prerequisits

Initialize OpenVINO™:
```
source $OV/setupvars.sh
```

Install the Model Optimizer prerequisites:
```
cd $OV/tools/model_optimizer/install_prerequisites/
sudo ./install_prerequisites.sh
```

Install the Model Downloader prerequisites:

```
cd $OV/extras/open_model_zoo/tools/downloader/
python3 -mpip install --user -r ./requirements.in
sudo python3 -mpip install --user -r ./requirements-pytorch.in
sudo python3 -mpip install --user -r ./requirements-caffe2.in
```

## 3. Download Models

Download all models from the Demo Models list:
```
python3 $OV/extras/open_model_zoo/tools/downloader/downloader.py --list $OV/deployment_tools/inference_engine/demos/python_demos/monodepth_demo/models.lst -o $WD
```

## 4. Convert Models to Intermediate Representation (IR)

Use the convert script to convert the models to ONNX*, and then to IR format:
```
cd $WD
python3 $OV/extras/open_model_zoo/tools/downloader/converter.py --list $OV/deployment_tools/inference_engine/demos/python_demos/monodepth_demo/models.lst
```

## 5. Run Demo

Install required Python modules, for example, kiwisolver or cycler, if you get missing module indication.

Use your input image:
```
python3 $OV/inference_engine/demos/python_demos/monodepth_demo/monodepth_demo.py -m $WD/public/midasnet/FP32/midasnet.xml -i input-image.jpg
```
Check the result depth image:
```
eog disp.png &
```
You can also try to use another model. Note that the algorithm is the same, but the depth map will be different. 
