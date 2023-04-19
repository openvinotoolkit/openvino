# AUTO Plugin Integration

## Implement a new plugin
Please refer to [OpenVINO Plugin Developer Guide](https://docs.openvino.ai/latest/openvino_docs_ie_plugin_dg_overview.html)

QueryNetwork(query_model) is recommanded since it is important for AUTO to make quick decision and save selection time.

## AUTO Plugin requirements of properties

.. note:: AUTO Plugin call IE APIs to plugins

| IE API                                       | OV API 2.0                           | Mandatory                             |
| -------------------------------------------- | ------------------------------------ | ------------------------------------- |
| CONFIG_KEY(DEVICE_ID)                        | ov::device::id                       | Yes                                   |
| CONFIG_KEY(PERF_COUNT)                       | ov::enable_profiling                 | Yes                                   |
| CONFIG_KEY(PERFORMANCE_HINT)                 | ov::hint::performance_mode           | Yes                                   |
| CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)    | ov::hint::num_requests               | Yes                                   |
| METRIC_KEY(FULL_DEVICE_NAME)                 | ov::device::full_name                | Yes                                   |
| METRIC_KEY(NETWORK_NAME)                     | ov::model_name                       | Yes                                   |
| METRIC_KEY(OPTIMAL_BATCH_SIZE)               | ov::optimal_batch_size               | No: Have this for size decision.      |
| METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) | ov::optimal_number_of_infer_requests | Yes                                   |
| METRIC_KEY(RANGE_FOR_STREAMS)                | ov::range_for_streams                | Yes                                   |
| METRIC_KEY(SUPPORTED_CONFIG_KEYS)            | ov::supported_properties             | Yes                                   |
| METRIC_KEY(SUPPORTED_METRICS)                | ov::supported_properties             | Yes                                   |
| METRIC_KEY(OPTIMIZATION_CAPABILITIES)        | ov::device::capabilities             | Yes                                   |
| METRIC_KEY(DEVICE_GOPS)                      | ov::device::gops                     | No: For automatic device selection priority. |
| METRIC_KEY(DEVICE_ARCHITECTURE)              | ov::device::architecture             | No                                    |
| METRIC_KEY(DEVICE_TYPE)                      | ov::device::type                     | No                                    |

For [Automatic Batching](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Automatic_Batching.html), following 2 properies should be supported.

| IE API                          | OV API 2.0                    | Mandatory |
| ------------------------------- | ----------------------------- | --------- |
| CONFIG_KEY(ALLOW_AUTO_BATCHING) | ov::hint::allow_auto_batching | Yes       |
| CONFIG_KEY(AUTO_BATCH_TIMEOUT)  | ov::auto_batch_timeout        | Yes       |

## AUTO Plugin tests
Please refer to [Testing the AUTO Plugin](./docs/tests.md)

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
| cumulative_throughput | AUTO:\<HW>,CPU |

### Test multiple devices running simultaneously
For example, there may be 2 GPUs on the same system, there will be GPU.0 GPU.1 devices

It is required for HW plugin to guarrantee multiple devices can be run in different threads simultaneously. It will be better if HW plugin can test with CPU plugin together by running them (CPU plugin and HW plugins) in different threads simultaneously.