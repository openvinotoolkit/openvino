# Face Detection Java Samples

This guide describes how to run the ```face_detection_java_sample``` and ```face_detection_sample_async``` applications.

## How It Works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates an output image/video.

To download model ( .bin and .xml files must be downloaded) use:
https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP32/

## Build
To create java library for Inference Engine add `-DENABLE_JAVA=ON` flag in cmake command while building dldt:
```bash
cd /openvino/build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_JAVA=ON ..
make
```

Create an environment variable with OpenCV installation path:
```bash
export OCV_PATH=/path/to/opencv/build
```

and the same for Inference Engine:
```bash
export IE_PATH=/path/to/openvino/bin/intel64/Release/lib
```

To compile parser class and sample use:
```bash
cd <sample_folder>
javac -cp ".:${OCV_PATH}/bin/*:${IE_PATH}/*" -d . ../ArgumentParser.java -d . Main.java
```

## Running
Add library path for openvino java library and for opencv library before running:
```bash
export LD_LIBRARY_PATH=${OCV_PATH}/lib:/${IE_PATH}:$LD_LIBRARY_PATH
```

To get sample help use:
```bash
java -cp ".:${OCV_PATH}/bin/*:${IE_PATH}/*" Main --help
```

Or to run sample use:
```bash
java -cp ".:${OCV_PATH}/bin/*:${IE_PATH}/*" Main <flags>
```

## Sample Output

### For ```face_detection_java_sample```
The application will show the image with detected objects enclosed in rectangles in new window. It outputs the confidence value and the coordinates of the rectangle to the standard output stream.

### For ```face_detection_sample_async```
The application saves a video ```result.avi``` with detected objects enclosed in rectangles.
