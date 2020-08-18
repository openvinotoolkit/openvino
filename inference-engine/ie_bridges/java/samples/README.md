# Benchmark Application

This guide describes how to run the benchmark applications.

## How It Works

Upon start-up, the application reads command-line parameters and loads a network to the Inference Engine plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend on the mode defined with the `-api` command-line parameter.

## Build
Create an environment variable with Inference Engine installation path:
export IE_PATH=/path/to/openvino/bin/intel64/Release/lib
```

To create java library and java samples for Inference Engine add `-DENABLE_JAVA=ON` flag in cmake command while building dldt:
```bash
cd /openvino/build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_JAVA=ON -DENABLE_SAMPLES=ON ..
make
```

## Running
Add library path for openvino java library before running:
```bash
export LD_LIBRARY_PATH=${IE_PATH}:$LD_LIBRARY_PATH
```

To get ```benchmark_app``` help use:
```bash
java -cp ".:${IE_PATH}/inference_engine_java_api.jar:${IE_PATH}/benchmark_app.jar" Main --help
```

To run ```benchmark_app`` use:
```bash
java -cp ".:${IE_PATH}/inference_engine_java_api.jar:${IE_PATH}/benchmark_app.jar" Main -m /path/to/model
```

## Application Output

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.

Below is fragment of application output for CPU device: 

```
[Step 10/11] Measuring performance (Start inference asyncronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
[Step 11/11] Dumping statistics report
Count:      8904 iterations
Duration:   60045.87 ms
Latency:    27.03 ms
Throughput: 148.29 FPS
```

# Face Detection Java Samples

## How It Works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates an output image/video.

To download model ( .bin and .xml files must be downloaded) use:
https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP32/

## Build and run

Build and run steps are similar to ```benchmark_app```, but you need to add OpenCV path.

### Build
Add an environment variable with OpenCV installation or build path:
```bash
export OpenCV_DIR=/path/to/opencv/
```

### Running
Add library path for opencv library and for openvino java library before running:

* For OpenCV installation path
```bash
export LD_LIBRARY_PATH=${OpenCV_DIR}/share/java/opencv4/:/${IE_PATH}:$LD_LIBRARY_PATH
```
To run sample use:
```bash
java -cp ".:${OpenCV_DIR}/share/java/opencv4/*:${IE_PATH}/inference_engine_java_api.jar:${IE_PATH}/sample_name.jar" Main -m /path/to/model -i /path/to/image
```

* For OpenCV build path
```bash
export LD_LIBRARY_PATH=${OpenCV_DIR}/lib:/${IE_PATH}:$LD_LIBRARY_PATH
```
To run sample use:
```bash
java -cp ".:${OpenCV_DIR}/bin/*:${IE_PATH}/inference_engine_java_api.jar:${IE_PATH}/sample_name.jar" Main -m /path/to/model -i /path/to/image
```

## Sample Output

### For ```face_detection_java_sample```
The application will show the image with detected objects enclosed in rectangles in new window. It outputs the confidence value and the coordinates of the rectangle to the standard output stream.

### For ```face_detection_sample_async```
The application will show the video with detected objects enclosed in rectangles in new window.
