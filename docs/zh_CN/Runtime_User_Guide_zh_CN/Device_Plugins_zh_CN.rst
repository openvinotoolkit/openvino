.. _openvino_docs_OV_UG_Working_with_devices_zh_CN:

推理设备支持
=====================================

.. toctree::
   :maxdepth: 1
   :hidden:

   CPU_zh_CN
   GPU_zh_CN


OpenVINO™ 运行时可以使用以下设备类型来推理深度学习模型：

* :doc:`CPU<CPU_zh_CN.rst>`
* :doc:`GPU<GPU_zh_CN.rst>`  
* `VPU<https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_VPU.html>`__
* `GNA<https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_GNA.html>`__ 
* `Arm® CPU<https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_ARM_CPU.html>`__  

有关更详细的硬件列表，请参见 `支持的设备 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html>`__

对于与我们用于基准测试的设备类似的设备，可以使用 `英特尔® DevCloud for the Edge <https://devcloud.intel.com/edge/>`__（一种可以访问英特尔® 硬件的远程开发环境）
和最新版本的英特尔® 发行版 OpenVINO™ 工具套件进行访问。 `了解更多信息 <https://devcloud.intel.com/edge/get_started/devcloud/>`__ 
或 `在此处注册 <https://inteliot.force.com/DevcloudForEdge/s/>`__。



功能支持表
#####################################

下表展示了 OpenVINO™ 器件插件支持的关键功能。

===================================================================================================== ===== ===== ===== ========== 
 功能                                                                                                    CPU   GPU   GNA   Arm® CPU  
===================================================================================================== ===== ===== ===== ========== 
 `异构执行 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Hetero_execution.html>`__                  是     是     否     是         
===================================================================================================== ===== ===== ===== ========== 
 `多设备执行 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Running_on_multiple_devices.html`__       是     是     部分    是         
===================================================================================================== ===== ===== ===== ========== 
 `自动批处理 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Automatic_Batching.html`__                否     是     否     否         
===================================================================================================== ===== ===== ===== ========== 
 `多流执行 <https://docs.openvino.ai/2022.3/openvino_docs_deployment_optimization_guide_tput.html`__       是     是     否     是         
===================================================================================================== ===== ===== ===== ========== 
 `模型缓存 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Model_caching_overview.html>`__            是     部分    是     否         
===================================================================================================== ===== ===== ===== ========== 
 `动态形状 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_DynamicShapes.html>`__                     是     部分    否     否         
===================================================================================================== ===== ===== ===== ========== 
 `导入/导出 <https://docs.openvino.ai/2022.3/openvino_inference_engine_tools_compile_tool_README.html`__   是     否     是     否         
===================================================================================================== ===== ===== ===== ========== 
 `预处理加速 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Preprocessing_Overview.html`__            是     是     否     部分        
===================================================================================================== ===== ===== ===== ========== 
 `有状态模型 <https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_network_state_intro.html`__               是     否     是     否         
===================================================================================================== ===== ===== ===== ========== 
 `扩展性 <https://docs.openvino.ai/2022.3/openvino_docs_Extensibility_UG_Intro.html`__                    是     是     否     否         
===================================================================================================== ===== ===== ===== ========== 


有关插件特定功能限制的更多详细信息，请参见相应的插件页面。

枚举可用设备
#####################################

OpenVINO™ 运行时 API 具有枚举设备及其功能的专用方法。请参阅 
`Hello 查询设备 C++ 样本 <https://docs.openvino.ai/2022.3/openvino_inference_engine_samples_hello_query_device_README.html>`__。
这是样本的示例输出（仅截断为设备名称）：


.. code-block:: sh

  ./hello_query_device
  Available devices:
      Device: CPU
  ...
      Device: GPU.0
  ...
      Device: GPU.1
  ...
      Device: HDDL


枚举设备并与多设备配合使用的简单编程方式如下：


.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI2.cpp
       :language: cpp
       :fragment: [part2]


除了典型的“CPU”、“GPU”、“HDDL”等之外，当设备的多个实例可用时，名称会更有限定性。例如，在 hello_query_sample 中这样枚举两个英特尔® Movidius™ Myriad™ X 电脑棒。

.. code-block:: sh
...
    Device: MYRIAD.1.2-ma2480
...
    Device: MYRIAD.1.4-ma2480


因此，使用这两者的显式配置将是“MULTI:MYRIAD.1.2-ma2480,MYRIAD.1.4-ma2480”。因此，循环遍历“MYRIAD”类型的所有可用设备的代码如下：



.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI3.cpp
       :language: cpp
       :fragment: [part3]
