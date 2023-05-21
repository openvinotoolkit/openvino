# AUTO Plugin Integration

## Implement a new plugin
Please refer to [OpenVINO Plugin Developer Guide](https://docs.openvino.ai/latest/openvino_docs_ie_plugin_dg_overview.html)

Query model method (`ov::IPlugin::query_model()`) is recommended since it is important for AUTO to make quick decision and save selection time.

## AUTO Plugin requirements of properties

.. note:: AUTO Plugin asks for the following plugin properties

| Property                             |  Manatory | Purpose                                       |
| ------------------------------------ |  -------- | --------------------------------------------- |
| ov::device::id                       |  Yes      | Distinguish devices with same type.           |
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
| ov::compilation_num_threads          |  No       | Limit the comilation threads for single device when compiling model to multiple devices. |

## AUTO Plugin tests
Please refer to [Testing the AUTO Plugin](./tests.md)

## AUTO Plugin integration tests

### Utilize benchmark_app to test AUTO together with hardware plugins

command: benchmark_app -d ${device} -hint ${hint} -m \<any model works on HW plugin>

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

### Test multiple devices running simultaneously
For example, there may be 2 GPUs on the same system, with devivce names as GPU.0 and GPU.1.

It is required for HW plugin to guarrantee multiple devices can be run in different threads simultaneously. It is recommended for HW plugin to test with CPU plugin, i.e., running CPU plugin and HW plugin in different threads simultaneously.