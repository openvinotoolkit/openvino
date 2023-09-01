# OpenVINO Benchmark Automation

The Batch class is a library for engineers to automate the running and logging of [OpenVINO Deep Learning Inference benchmark tests](https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html). To use the Batch class, please follow these four simple steps:<br/>

 1. <pre>pip install -r requirements.txt</pre>
 2. Set the environment variable <b>OPENVINO_PATH</b> to the folder where OpenVINO is installed on your machine. On Linux machines this will usually be <b>"/opt/intel/openvino_2022"</b> but with your specific version. This is necessary for initializing the OpenVINO environment, otherwise all benchmarks will fail.
 3. <pre>python3 main.py [MODEL_FOLDER] [CONFIG_FILE]</pre> Replace <b>MODEL_FOLDER</b> with the path of your machine's OpenVINO model folder and <b>CONFIG_FILE</b> with the path to a JSON file containing your benchmark settings. On Linux machines the model folders will usually be <b>"~/public"</b> or <b>"~/intel"</b>. You can find examples for both in this repository in the <b>example_model</b> folder and <b>example_settings.json</b> file. If these two arguments are omitted in the command line, it will automatically select the example paths.
 4. Open the output folder to view a timestamped subfolder containing all benchmark results, including full text logs and a summary CSV.
<br/><br/><br/>
This library was developed by Henry Tsai, Software Engineering Intern at Intel Automotive Chandler, in July 2023 under the guidance of Rony Ferzli and Keith Perrin. The original usage for the library was to benchmark face detection models for automotive applications. As such, there is a Jupyter Notebook included containing data analysis of selected face detection model results which can be accessed with the terminal command <b>jupyter notebook</b> and opening the resulting window.
<br/><br/><br/>
Future development of this library could include:
* Adding support for additional benchmark app arguments in the benchmark method
* Adding support for benchmarking .onnx model files (currently only supports native OpenVINO .xml files)
* Adding functionality for automatically converting non-native model files before running benchmarks using the [OpenVINO Model Converter](https://docs.openvino.ai/2023.0/omz_tools_downloader.html#model-converter-usage)
* Improving error handling for CSV summary generation on error logs
* Including system information (e.g. Linux lscpu/clinfo commands) in logs
