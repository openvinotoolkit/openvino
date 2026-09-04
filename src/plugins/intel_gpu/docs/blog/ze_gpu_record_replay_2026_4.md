# Support for Level Zero command list record and replay: New in OpenVINO 2026.4

*By Jakub Kasprzak | September 1, 2026*

Sometimes, preparing and submitting a GPU kernel can take more time than actually executing it. Running an AI model consisting of many such kernels can result in GPU underutilization as the CPU is not able to submit work fast enough and becomes the bottleneck. OpenVINO 2026.4 introduces GPU record and replay feature based on Level Zero command list. It can improve average latency of AI models that are executed more than once by enabling OpenVINO to replay GPU commands from previous iteration - preventing situations in which the GPU remains idle while waiting for CPU to submit next kernel.

## How to enable Level Zero command list record and replay

By default GPU record and replay is disabled and user must set `GPU_RECORD_REPLAY` option to enable it for supported static models. Additional debug option `GPU_RECORD_REPLAY_DYNAMIC_WIP` enables the feature on all models but may produce incorrect results.

Learn how to set OpenVINO options here `src/plugins/intel_gpu/docs/gpu_debug_utils.md`.
