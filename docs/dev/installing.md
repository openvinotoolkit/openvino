# Installing

Once the project is built you can install OpenVINO™ Runtime into custom location:
 
```
cmake --install <BUILDDIR> --prefix <INSTALLDIR>
```

## Installation check

<details>
<summary>For versions prior to 2022.1</summary>
<p>

1. Obtaining Open Model Zoo tools and models

To have the ability to run samples and demos, you need to clone the Open Model Zoo repository and copy the folder under `./deployment_tools` to your install directory:

```
git clone https://github.com/openvinotoolkit/open_model_zoo.git
cmake -E copy_directory ./open_model_zoo/ <INSTALLDIR>/deployment_tools/open_model_zoo/
```

2. Adding OpenCV to your environment

Open Model Zoo samples use OpenCV functionality to load images. To use it for demo builds you need to provide the path to your OpenCV custom build by setting `OpenCV_DIR` environment variable and add path OpenCV libraries to the `LD_LIBRARY_PATH (Linux)` or `PATH (Windows)` variable before running demos.

Linux:
```sh
export LD_LIBRARY_PATH=/path/to/opencv_install/lib/:$LD_LIBRARY_PATH
export OpenCV_DIR=/path/to/opencv_install/cmake
```

Windows:
```sh
set PATH=\path\to\opencv_install\bin\;%PATH%
set OpenCV_DIR=\path\to\opencv_install\cmake
```

3. Running demo

To check your installation go to the demo directory and run Classification Demo:

Linux and macOS:
```sh
cd <INSTALLDIR>/deployment_tools/demo
./demo_squeezenet_download_convert_run.sh
```

Windows:
```sh
cd <INSTALLDIR>\deployment_tools\demo
demo_squeezenet_download_convert_run.bat
```

Result:
```
Top 10 results:

Image <INSTALLDIR>/deployment_tools/demo/car.png

classid probability label
------- ----------- -----
817     0.6853030   sports car, sport car
479     0.1835197   car wheel
511     0.0917197   convertible
436     0.0200694   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
751     0.0069604   racer, race car, racing car
656     0.0044177   minivan
717     0.0024739   pickup, pickup truck
581     0.0017788   grille, radiator grille
468     0.0013083   cab, hack, taxi, taxicab
661     0.0007443   Model T

[ INFO ] Execution successful
```

</p>
</details>


<details open>
<summary> For 2022.1 and after</summary>
<p>

1. Build samples

To build C++ sample applications, run the following commands:

Linux and macOS:
```sh
cd <INSTALLDIR>/samples/cpp
./build_samples.sh
```

Windows:
```sh
cd <INSTALLDIR>\samples\cpp
build_samples_msvc.bat
```

2. Install OpenVINO Development Tools

> **NOTE**: To build OpenVINO Development Tools (Model Optimizer, Post-Training Optimization Tool, Model Downloader, and Open Model Zoo tools) wheel package locally you are required to use the CMake option: `-DENABLE_WHEEL=ON`.

To install OpenVINO Development Tools to work with Caffe models (OpenVINO support for Caffe is currently being deprecated and will be removed entirely in the future), execute the following commands:

Linux and macOS:

```sh
#setup virtual environment
python3 -m venv openvino_env
source openvino_env/bin/activate
pip install pip --upgrade

#install local package from install directory
pip install openvino_dev-<version>-py3-none-any.whl[caffe]  --find-links=<INSTALLDIR>/tools
```

Windows:
```bat
rem setup virtual environment
python -m venv openvino_env
openvino_env\Scripts\activate.bat
pip install pip --upgrade

rem install local package from install directory
cd <INSTALLDIR>\tools
pip install openvino_dev-<version>-py3-none-any.whl[caffe] --find-links=<INSTALLDIR>\tools
```

3.  Download the Models

Download the following model to run the Image Classification Sample:

Linux and macOS:
```sh
omz_downloader --name googlenet-v1 --output_dir ~/models
```

Windows:
```bat
omz_downloader --name googlenet-v1 --output_dir %USERPROFILE%\Documents\models
```

4. Convert the Model with Model Optimizer

Linux and macOS:
```sh
mkdir ~/ir
mo --input_model ~/models/public/googlenet-v1/googlenet-v1.caffemodel --compress_to_fp16 --output_dir ~/ir
```
Windows:
```bat
mkdir %USERPROFILE%\Documents\ir
mo --input_model %USERPROFILE%\Documents\models\public\googlenet-v1\googlenet-v1.caffemodel --compress_to_fp16 --output_dir %USERPROFILE%\Documents\ir
```

5. Run Inference on the Sample

Set up the OpenVINO environment variables:

Linux and macOS:
```sh
source <INSTALLDIR>/setupvars.sh
```

Windows:
```bat
<INSTALLDIR>\setupvars.bat
```

The following commands run the Image Classification Code Sample using the [`dog.bmp`](https://storage.openvinotoolkit.org/data/test_data/images/224x224/dog.bmp) file as an input image, the model in IR format from the `ir` directory, and on different hardware devices:

Linux and macOS:

```sh
cd ~/openvino_cpp_samples_build/<architecture>/Release
./classification_sample_async -i ~/Downloads/dog.bmp -m ~/ir/googlenet-v1.xml -d CPU
```
where the <architecture> is the output of ``uname -m``, for example, ``intel64``, ``armhf``, or ``aarch64``.

Windows:

```bat
cd  %USERPROFILE%\Documents\Intel\OpenVINO\openvino_cpp_samples_build\<architecture>\Release
.\classification_sample_async.exe -i %USERPROFILE%\Downloads\dog.bmp -m %USERPROFILE%\Documents\ir\googlenet-v1.xml -d CPU
```
where the <architecture> is either ``intel64`` or ``aarch64`` depending on the platform architecture.

When the sample application is complete, you see the label and confidence data for the top 10 categories on the display:

```
Top 10 results:

Image dog.bmp

classid probability
------- -----------
156     0.6875963
215     0.0868125
218     0.0784114
212     0.0597296
217     0.0212105
219     0.0194193
247     0.0086272
157     0.0058511
216     0.0057589
154     0.0052615

```

</p>
</details>

## Adding OpenVINO Runtime (Inference Engine) to Your Project

<details>
<summary>For versions prior to 2022.1</summary>
<p>

For CMake projects, set the `InferenceEngine_DIR` and when you run CMake tool:

```sh
cmake -DInferenceEngine_DIR=/path/to/openvino/build/ .
```

Then you can find Inference Engine by [`find_package`]:

```cmake
find_package(InferenceEngine REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES})
```
</p>
</details>


<details open>
<summary>For 2022.1 and after</summary>
<p>


For CMake projects, set the `OpenVINO_DIR` and when you run CMake tool:

```sh
cmake -DOpenVINO_DIR=<INSTALLDIR>/runtime/cmake .
```

Then you can find OpenVINO Runtime (Inference Engine) by [`find_package`]:

```cmake
find_package(OpenVINO REQUIRED)
add_executable(ov_app main.cpp)
target_link_libraries(ov_app PRIVATE openvino::runtime)

add_executable(ov_c_app main.c)
target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)
```
</p>
</details>

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO How to Build](build.md)

 