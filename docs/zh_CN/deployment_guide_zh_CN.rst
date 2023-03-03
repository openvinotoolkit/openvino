.. _openvino_deployment_guide_zh_CN:

使用 OpenVINO™ 部署应用
============================

完成 `OpenVINO™ 应用开发 <https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_Integrate_OV_with_your_application.html>`_
后，应用开发人员通常需要为最终用户部署应用。有几种方法可以实现这一目标：

- 设置对现有预构建程序包的依赖项（也称为“集中分发”）：
  
   - 使用 Debian/RPM 程序包 - 建议 Linux 操作系统采用的方法；
   - 在 PyPI 上使用 PIP 程序包管理器 - 基于 Python 的应用的默认方法；
   - 使用 Docker 映像 - 如果应将应用部署为 Docker 映像，则将预构建的 OpenVINO™ 运行时 Docker 映像作为 Dockerfile 中应用容器映像的基础映像。
      有关 OpenVINO™ Docker 映像的更多信息，请参阅 `在 Linux 上从 Docker 安装 OpenVINO™ <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_docker_linux.html>`_
      和 `在 Windows 上从 Docker 安装 OpenVINO <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_docker_windows.html>`_。
      另外，如需自定义 OpenVINO™ Docker 映像，请使用 `Docker CI 框架 <https://github.com/openvinotoolkit/docker_ci>`_ 生成 Dockerfile 并构建映像。

- 获取 OpenVINO™ 以及您的应用的必要功能（也称为“本地分发”）：
   - 使用 `OpenVINO™ 部署管理器 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_deployment_manager_tool.html>`_ - 用于轻松创建分发程序包；
   - 使用高级 `本地分发 <https://docs.openvino.ai/2022.2/openvino_docs_deploy_local_distribution.html>`_ 方法；
   - 使用 `链接到最终应用的静态版本 OpenVINO™ 运行时 <https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries>`_。

下表显示了不同分发类型适用的不同目标操作系统：

.. list-table::
   :header-rows: 1

   * - 分发类型
     - 操作系统
   * - Debian 程序包
     - Ubuntu 18.04 长期支持 (LTS)，64 位；Ubuntu 20.04 长期支持 (LTS)，64 位 
   * - RMP 程序包
     - Red Hat Enterprise Linux 8，64 位
   * - Docker 映像
     - Ubuntu 18.04 长期支持 (LTS)，64 位；Ubuntu 20.04 长期支持 (LTS)，64 位；Red Hat Enterprise Linux 8，64 位；Windows Server Core base LTSC 2019，64 位；Windows 10，20H2 版，64 位                    
   * - PyPI（PIP 程序包管理器）
     - 请参阅 `pypi.org/project/openvino/ <https://pypi.org/project/openvino/>`_          
   * - `OpenVINO™ 部署管理器 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_deployment_manager_tool.html>`_
     - 所有操作系统                    
   * - `本地分发 <https://docs.openvino.ai/2022.2/openvino_docs_deploy_local_distribution.html>`_
     - 所有操作系统
   * - `静态构建 OpenVINO™ 并链接到最终应用 <https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries>`_
     - 所有操作系统


主要分发类型的粒度
------------------------------

根据分发类型的不同，OpenVINO™ 程序包的粒度可能会有所不同。例如，OpenVINO™ 的 PyPI 分发具有
`单一“openvino”程序包 <https://pypi.org/project/openvino/>`_。其中包含所有运行时库和插件，而
`本地分发 <https://docs.openvino.ai/2022.2/openvino_docs_deploy_local_distribution.html>`_
类型的可配置性更强，因而具有更高的粒度。以下是 OpenVINO™ 运行时程序包中包含的库集合的重要详细信息：

.. image:: assets/deployment_simplified.svg


- 主库 `openvino` 用于用户的 C++ 应用并与其链接在一起。该库提供所有 OpenVINO™ 运行时公共 API，包括 API 2.0 和以前的推理引擎以及 nGraph API。对于 C 语言应用，分发时还需要 `openvino_c`。
- `openvino_intel_cpu_plugin` 等“可选”插件库（匹配 `openvino_.+_plugin` 模式）用于在特定设备上提供推理功能，或提供附加功能，如 `异构执行 <https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_Hetero_execution.html>`_ 和 `多设备执行 <https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_Running_on_multiple_devices.html>`_。
- `openvino_ir_frontend` 等“可选”插件库（匹配 `openvino_.+_frontend`）用于提供多种功能来读取 OpenVINO™ IR、ONNX 和 PaddlePaddle 等不同文件格式的模型。

此处“可选”是指如果应用不使用插件启用的功能，最终分发中将无需插件库或包含插件的程序包。

构建本地分发将需要更多详细信息，您可在专用文章 `本地分发所需的库 <https://docs.openvino.ai/2022.2/openvino_docs_deploy_local_distribution.html>`_ 中找到这些信息。

.. note:: 
   根据您的目标 OpenVINO™ 器件，已部署机器可能需要以下配置：
   `针对 GPU 的配置 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_configurations_for_intel_gpu.html>`_、
   `针对 GNA 的配置 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_configurations_for_intel_gna.html>`_、
   `针对 NCS2 的配置 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_configurations_for_ncs2.html>`_、
   `针对 VPU 的配置 <https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_ivad_vpu.html>`_。