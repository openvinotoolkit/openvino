# Intel® Movidius™ VPU Demos for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_movidius_demos}

@sphinxdirective

.. _vpu demos:

@endsphinxdirective

Once you have your Intel® Distribution of OpenVINO™ Toolkit installed, and configured your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs [Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Configuration Guide](installing-openvino-linux-ivad-vpu.md), you can run our demos:

1. Go to the **Inference Engine demo** directory:
   ```sh
   cd <INSTALL_DIR>/intel/openvino_2022/deployment_tools/demo
   ```

2. Run the **Image Classification verification script**. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.
   ```sh
   ./demo_squeezenet_download_convert_run.sh -d HDDL
   ```

3. Run the **Inference Pipeline verification script**:
   ```sh
   ./demo_security_barrier_camera.sh -d HDDL
   ```

You've completed all required configuration steps to perform inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
