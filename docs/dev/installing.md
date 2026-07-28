# Installing

Once the project is built you can install OpenVINO™ Runtime into custom location:

```
cmake --install <BUILDDIR> --prefix <INSTALLDIR>
```

## Build and Run Samples

1. Build samples.

   To build C++ sample applications, run the following commands:

   Linux and macOS:
   ```sh
   cd <INSTALLDIR>/samples/cpp
   ./build_samples.sh
   ```

   Windows Command Prompt:
   ```sh
   cd <INSTALLDIR>\samples\cpp
   build_samples_msvc.bat
   ```

   Windows PowerShell:
   ```sh
   & <path-to-build-samples-folder>/build_samples.ps1
   ```

2. Download a model.

   You can download an image classification model from
   [Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)
   to run the sample

4. Convert the model.

   Linux and macOS:
   ```sh
   ovc <path-to-your-model> --compress_to_fp16=True
   ```
   Windows:
   ```bat
   ovc <path-to-your-model> --compress_to_fp16=True
   ```

5. Run inference on the sample.

   Set up the OpenVINO environment variables:

   Linux and macOS:
   ```sh
   source <INSTALLDIR>/setupvars.sh
   ```

   Windows Command Prompt:
   ```bat
   <INSTALLDIR>\setupvars.bat
   ```

   Windows PowerShell:
   ```bat
   . <path-to-setupvars-folder>/setupvars.ps1
   ```

   The following commands run the Image Classification Code Sample using the [`dog.bmp`](https://storage.openvinotoolkit.org/data/test_data/images/   224x224/dog.bmp) file as an input image, the model in IR format, and on different hardware devices:

   Linux and macOS:

   ```sh
   cd ~/openvino_cpp_samples_build/<architecture>/Release
   ./classification_sample_async -i <path-to-input-image>/dog.bmp -m <path-to-your-model>/model.xml -d CPU
   ```
   where the <architecture> is the output of ``uname -m``, for example, ``intel64``, ``armhf``, or ``aarch64``.

   Windows:

   ```bat
   cd  %USERPROFILE%\Documents\Intel\OpenVINO\openvino_cpp_samples_build\<architecture>\Release
   .\classification_sample_async.exe -i <path-to-input-image>\dog.bmp -m <path-to-your-model>\model.xml -d CPU
   ```
   where the <architecture> is either ``intel64`` or ``aarch64`` depending on the platform architecture.

When the sample application is complete, you see the label and confidence data for the top 10 categories on the display:

Below are results of using the googlenet-v1 model.

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


## Adding OpenVINO Runtime to Your Project

For CMake projects, set the `OpenVINO_DIR` and when you run CMake tool:

```sh
cmake -DOpenVINO_DIR=<INSTALLDIR>/runtime/cmake .
```

Then you can find OpenVINO Runtime by [`find_package`]:

```cmake
find_package(OpenVINO REQUIRED)
add_executable(ov_app main.cpp)
target_link_libraries(ov_app PRIVATE openvino::runtime)

add_executable(ov_c_app main.c)
target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)
```

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO How to Build](build.md)

