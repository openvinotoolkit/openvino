# [Experimental] Level Zero command list record and replay

Sometimes, preparing and submitting a GPU kernel can take more time than actually executing it. Running an AI model consisting of many such kernels can result in GPU underutilization as the CPU is not able to submit work fast enough and becomes the bottleneck. Record and replay feature based on Level Zero command list can improve average performance of such models by recording GPU commands during first iteration and then replaying them on subsequent iterations.

Please note that this is an internal option currently, and support in the future is not guaranteed.

## How to enable Level Zero command list record and replay

By default GPU record and replay is disabled and user must set `OV_GPU_RECORD_REPLAY=1` environment variable to enable it for supported static models. Additional debug option `OV_GPU_RECORD_REPLAY_DYNAMIC=1` enables the feature on all models but may produce incorrect results and requires OpenVINO build with `ENABLE_DEBUG_CAPS=ON`.

Learn how to set OpenVINO options here `src/plugins/intel_gpu/docs/gpu_debug_utils.md`.

## What models are supported

Currently only some static models are supported by the record and replay feature. When network created for the model contains primitives that does not support replay then exception is thrown. Please check `supports_replay()` in the primitives.

Support check can be skipped by setting `OV_GPU_RECORD_REPLAY_DYNAMIC=1` option. This option is added to understand performance status with command list.

## How recording works

Recording iteration can be divied into 2 phases.

1. Capture phase

    During this phase GPU commands are not submitted to the device but instead they are captured to Level Zero command list. Once all network primitives are processed OpenVINO advances to submit phase. This phase can be interrupted when executing dynamic network and any of the executed primitives requires GPU to host synchronization. When interrupted, all commands captured up to this point are submitted, the remaining primitives are executed with immediate command list and re-recording is attempted on the next iteration.

2. Submit phase

    During this phase all captured commands are submitted to the device at once and the network enqueue concludes.

Inference latency during recording (capture phase + submit phase + wait) can be up to 2 times longer than immediate execution.

## How replay works

During replay iteration OpenVINO submits all recorded commands at once and skips the usual primitive preparation and execution phases performed on CPU - preventing situations in which the GPU remains idle while waiting for CPU to submit next kernel.

Inference latency during replay (submit phase + wait) should not be worse than immediate execution.

## Recording invalidation

Command list is recorded at network level and captures input and output memory pointers. Changing input or output of the network will cause recording invalidation and forces network to record again on the next iteration. Repeated re-recording will result in performance degradation and for such use cases it is advised to disable this feature.

## Multiple inference requests

Using record and replay feature with more than 1 inference request can lead to recording invalidation as inference requests can bind to any available stream and update the network.
