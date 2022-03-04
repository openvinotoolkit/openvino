# Post-Training Optimization Tool - A real example

This tutorial describes the example from the following YouTube* video:
https://www.youtube.com/watch?v=cGQesbWuRhk&t=49s


Watch this video to learn the basics of Post-training Optimization Tool (POT): 
   https://www.youtube.com/watch?v=SvkI25Ca_SQ   

The example has been tested on OpenVINO™ 2021 on Ubuntu 18.04 Operating System.


## 1. Installation

Install OpenVINO™ toolkit and Model Optimizer, Accuracy Checker, and Post-training Optimization Tool components.

1. Define the OpenVINO™ install directory:
```
export OV=/opt/intel/openvino_2022/
```
2. Install the Model Optimizer prerequisites:
```
cd $OV/tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites.sh
```
3. Install the Accuracy Checker requirements:
```
cd $OV/tools/accuracy_checker
sudo python3 setup.py install
```
4. Install the Post-training Optimization Tool:
```
cd $OV/tools/post_training_optimization_toolkit
sudo python3 setup.py install
```

## 2. Download Model

This tutorial describes MobileNet v2 model from PyTorch* framework. You can choose any other model. 

Download the MobileNet v2 PyTorch* model using the commands below:
```
mkdir ~/POT
```
```
cd ~/POT
```
```
python3 $OV/extras/open_model_zoo/tools/downloader/downloader.py --name mobilenet-v2-pytorch -o .
```

## 3. Prepare Model for Inference

Install requirements for PyTorch using the commands below:
```
cd $OV/extras/open_model_zoo/tools/downloader
```
```
python3 -mpip install --user -r ./requirements-pytorch.in
```

You can find the parameters for Mobilnet v2 conversion here:
```
vi /opt/intel/openvino_2022/extras/open_model_zoo/models/public/mobilenet-v2-pytorch/model.yml
```

Convert the model from PyTorch to ONNX*:
```
cd ~/POT/public/mobilenet-v2-pytorch
python3 /opt/intel/openvino_2022/extras/open_model_zoo/tools/downloader/pytorch_to_onnx.py  \
    --model-name=MobileNetV2 \
    --model-path=.  \
    --weights=mobilenet-v2.pth \
    --import-module=MobileNetV2  \
    --input-shape=1,3,224,224 /
    --output-file=mobilenet-v2.onnx  \
    --input-names=data  \
    --output-names=prob

```
Convert the model from ONNX to the OpenVINO™ Intermediate Representation (IR):
```
mo  \
    -m mobilenet-v2.onnx \  
    --input=data  \
    --mean_values=data[123.675,116.28,103.53]  \
    --scale_values=data[58.624,57.12,57.375]  \
    --reverse_input_channels        \
    --output=prob  
```

Move the IR files to my directory:

```
mv mobilenet-v2.xml ~/POT/model.xml
mv mobilenet-v2.bin ~/POT/model.bin
```

## 4. Edit Configurations 

Edit the configuration files:
```
sudo vi $OV/tools/accuracy_checker/dataset_definitions.yml
(edit imagenet_1000_classes)
```
```
export DEFINITIONS_FILE=/opt/intel/openvino_2022/tools/accuracy_checker/dataset_definitions.yml
```

Copy the JSON file to my directory and edit:

```
cp $OV/tools/post_training_optimization_toolkit/configs/examples/quantization/classification/mobilenetV2_pytorch_int8.json ~/POT
```
```
vi mobilenetV2_pytorch_int8.json
```

Copy the YML file to my directory and edit:

```
cp /opt/intel/openvino_2022/tools/accuracy_checker/configs/mobilenet-v2.yml ~/POT
```
```
vi mobilenet-v2.yml
```

## 5. Run Baseline 

Run Accuracy Checker on the original model:

```
accuracy_check -c mobilenet-v2.yml
```

Install the Benchmark Tool first. To learn more about Benchmark Tool refer to [Benchmark C++ Tool](https://docs.openvino.ai/latest/openvino_inference_engine_samples_benchmark_app_README.html)
 or [Benchmark Python* Tool](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html).

Run performance benchmark:
```
~/inference_engine_cpp_samples_build/intel64/Release/benchmark_app -m ~/POT/model.xml
```

## 6. Run Integer Calibration

You can edit the JSON file to switch between two modes of calibration:

 -  AccuracyAwareQuantization
 -  DefaultQuantization


```
pot --config      /home/~/POT/mobilenetV2_pytorch_int8.json   \
        --output-dir /home/~/POT/        \ 
        --evaluate                            \
        --log-level INFO 
```

Run the Benchmark Tool for the calibrated model. Make sure the name contains `DafultQuantization/.../optimized/...`

```
~/inference_engine_cpp_samples_build/intel64/Release/benchmark_app -m mobilenetv2_DefaultQuantization/2021-03-07/optimized/mobilenetv2.xml
```
