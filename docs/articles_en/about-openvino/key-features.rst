Key Features
==============

Easy Integration
#########################

| :doc:`Support for multiple frameworks <../openvino-workflow/model-preparation/convert-model-to-ir>`
|     Use deep learning models from PyTorch, TensorFlow, TensorFlow Lite, PaddlePaddle, and ONNX
      directly or convert them to the optimized OpenVINO IR format for improved performance.

| :doc:`Close integration with PyTorch <../openvino-workflow/torch-compile>`
|     For PyTorch-based applications, specify OpenVINO as a backend using
      :doc:`torch.compile <../openvino-workflow/torch-compile>` to improve model inference. Apply
      OpenVINO optimizations to your PyTorch models directly with a single line of code.

| :doc:`GenAI Out Of The Box <../learn-openvino/llm_inference_guide/genai-guide>`
|     With the genAI flavor of OpenVINO, you can run generative AI with just a couple lines of code.
      Check out the GenAI guide for instructions on how to do it.

| `Python / C++ / C / NodeJS APIs <https://docs.openvino.ai/2024/api/api_reference.html>`__
|     OpenVINO offers the C++ API as a complete set of available methods. For less resource-critical
      solutions, the Python API provides almost full coverage, while C and NodeJS ones are limited
      to the methods most basic for their typical environments. The NodeJS API, is still in its
      early and active development.

| :doc:`Open source and easy to extend <../about-openvino/contributing>`
|     If you need a particular feature or inference accelerator to be supported, you are free to file
      a feature request or develop new components specific to your projects yourself. As open source,
      OpenVINO may be used and modified freely. See the extensibility guide for more information on
      how to adapt it to your needs.

Deployment
#########################

| :doc:`Local or remote <../openvino-workflow>`
|     Integrate the OpenVINO runtime directly with your application to run inference locally or use
      `OpenVINO Model Server <https://github.com/openvinotoolkit/model_server>`__ to shift the inference
      workload to a remote system, a separate server or a Kubernetes environment. For serving,
      OpenVINO is also integrated with `vLLM <https://docs.vllm.ai/en/stable/getting_started/openvino-installation.html>`__
      and `Triton <https://github.com/triton-inference-server/openvino_backend>`__ services.

| :doc:`Scalable and portable <release-notes-openvino/system-requirements>`
|     Write an application once, deploy it anywhere, always making the most out of your hardware setup.
      The automatic device selection mode gives you the ultimate deployment flexibility on all major
      operating systems. Check out system requirements.

| **Light-weight**
|     Designed with minimal external dependencies, OpenVINO does not bloat your application
      and simplifies installation and dependency management. The custom compilation for your specific
      model(s) may further reduce the final binary size.

Performance
#########################

| :doc:`Model Optimization <../openvino-workflow/model-optimization>`
|     Optimize your deep learning models with NNCF, using various training-time and post-training
      compression methods, such as pruning, sparsity, quantization, and weight compression. Make
      your models take less space, run faster, and use less resources.

| :doc:`Top performance <../about-openvino/performance-benchmarks>`
|     OpenVINO is optimized to work with Intel hardware, delivering confirmed high performance for
      hundreds of models. Explore OpenVINO Performance Benchmarks to discover the optimal hardware
      configurations and plan your AI deployment based on verified data.

| :doc:`Enhanced App Start-Up Time <../openvino-workflow/running-inference/optimize-inference>`
|     If you need your application to launch immediately, OpenVINO will reduce first-inference latency,
      running inference on CPU until a more suited device is ready to take over. Once a model
      is compiled for inference, it is also cached, improving the start-up time even more.

