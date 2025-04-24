# AUTO Plugin Integration

## Implement a New Plugin
Refer to [OpenVINO Plugin Developer Guide](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html) for detailed information on how to implement a new plugin.

Query model method `ov::IPlugin::query_model()` is recommended as it is important for AUTO to quickly make decisions and save selection time.

## AUTO Plugin Property Requirements

AUTO Plugin requires the following plugin properties:

| Property                             |  Mandatory | Purpose                                       |
| ------------------------------------ |  -------- | --------------------------------------------- |
| ov::device::id                       |  Yes      | Distinguish devices with the same type.           |
| ov::enable_profiling                 |  Yes      | Performance profiling.                        |
| ov::hint::performance_mode           |  Yes      | Performance mode hint.                        |
| ov::hint::num_requests               |  Yes      | num_requests hint.                            |
| ov::device::full_name                |  Yes      | Automatic device selection.                   |
| ov::model_name                       |  Yes      | Return model name.                            |
| ov::optimal_batch_size               |  No       | Decide batch size in automatic batching case. |
| ov::optimal_number_of_infer_requests |  Yes      | Decide AUTO optimal_number_of_infer_requests. |
| ov::range_for_streams                |  Yes      | Decide AUTO optimal_number_of_infer_requests in automatic batching case. |
| ov::supported_properties             |  Yes      | Check if a property is supported by HW plugin.|
| ov::device::capabilities             |  Yes      | Automatic device selection.                   |
| ov::device::gops                     |  No       | Improve automatic device selection.           |
| ov::compilation_num_threads          |  No       | Limit the compilation threads for a single device when compiling a model to multiple devices. |

## AUTO Plugin Tests
Refer to the [Testing the AUTO Plugin](./tests.md) page for detailed instructions.

## AUTO Plugin Integration Tests

### Test AUTO and Hardware Plugins Using benchmark_app

```sh
benchmark_app -d ${device} -hint ${hint} -m <any model works on HW plugin>
```

| hint                  | device        |
| --------------------- | ------------- |
| throughput            | \<HW>          |
| throughput            | AUTO:\<HW>     |
| throughput            | AUTO:\<HW>,CPU |
| latency               | \<HW>          |
| latency               | AUTO:\<HW>     |
| latency               | AUTO:\<HW>,CPU |
| cumulative_throughput | AUTO:\<HW>     |
| cumulative_throughput | AUTO:\<HW>,CPU |

### Test Multiple Devices Running Simultaneously

The HW plugin must guarantee simultaneous execution of multiple devices in different threads. It is recommended to test the HW plugin with the CPU plugin by running the plugins in different threads simultaneously.

For example, there may be two GPUs on the same system, with device names GPU.0 and GPU.1. GPU plugin must guarantee simultaneous execution of GPU.0 and GPU.1 in different threads.