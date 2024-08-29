Key Features
==============

| **Support for multiple frameworks**
| Use deep learning models from PyTorch, TensorFlow, TensorFlow Lite, PaddlePaddle, and ONNX
  directly or convert them to the optimized OpenVINO IR format for improved performance.

| **Close integration with PyTorch**
| For PyTorch-based applications, specify OpenVINO as a backend using
  :doc:`torch.compile <../openvino-workflow/torch-compile>` to improve model inference. Apply
  OpenVINO optimizations to your PyTorch models directly with a single line of code.

| **Model Optimization**
| Optimize your deep learning models with :doc:`NNCF <../openvino-workflow/model-optimization>`,
  using various training-time and post-training compression methods. Make Your models take
  less space, run faster, and use less resources.

.. Optimize your deep learning models with NNCF, by reducing model size and accelerating inference.
.. NNCF provides different optimization techniques, including post-training quantization without retraining,
.. and training-time optimizations like Quantization-aware Training, Pruning and Sparsity. For large language models,
.. NNCF offers weight compression method to decrease model footprint.

| **Top performance**
| OpenVINO is optimized to work with Intel hardware, delivering confirmed high performance for
  hundreds of models. Explore OpenVINO Performance Benchmarks to discover the optimal hardware
  configurations and plan your AI deployment based on verified data.

| **Deploy locally or use serving**
| Integrate the OpenVINO runtime directly with your application to run inference locally or use
  `OpenVINO Model Server <https://github.com/openvinotoolkit/model_server>`__ to shift the inference
  workload to a remote system, a separate server or a Kubernetes environment. For serving,
  OpenVINO is also integrated with vLLM and Triton services.

| **Scalable and portalbe - automatic device Selection**
| Write an application once, deploy it anywhere, always making the most out of your hardware setup.
  The automatic device selection mode gives you the ultimate deployment flexibility, on all major
  operating systems (check out :doc:`system requirements <release-notes-openvino/system-requirements>`).

| **Lighter Deployment**
| Designed with minimal external dependencies, OpenVINO does not bloat your application.
| Simplifying installation and dependency management. Popular package managers enable application
  dependencies to be easily installed and upgraded. Custom compilation for your specific model(s)
  further reduces the final binary size.

| **Python / C++ / C / NodeJS APIs**
| As the main OpenVINO API, the C++ one offers the complete set of available operations. The Python
  API provides an almost complete mapping of it, while C and NodeJS provide a very limited
  functionality. JS is still in its early development stages.

| **OpenVINO Extensibility**
| You can develop your own solutions for OV easly - open source.

| **Enhanced App Start-Up Time**
| In applications where fast start-up is required, OpenVINO significantly reduces first-inference
  latency by using the CPU for initial inference and then switching to another device once
  the model has been compiled and loaded to memory. Compiled models are cached, improving start-up
  time even more.

| **GenAI Out Of The Box**
| With the :doc:`GenAI flavor <../learn-openvino/llm_inference_guide/genai-guide>` of OpenVINO,
  run generative AI withe just couple of lines of code. With full scope available in the
  :doc:`LLM Inference Guide <../learn-openvino/llm_inference_guide>`.

