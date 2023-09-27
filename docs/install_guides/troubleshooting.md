# Troubleshooting Guide for OpenVINO™ Installation & Configuration {#openvino_docs_get_started_guide_troubleshooting}

@sphinxdirective

.. meta::
   :description: A collection of troubleshooting steps and solutions to possible 
                 problems that may occur during the installation and configuration 
                 of OpenVINO™ on your system.


.. _troubleshooting guide for install:

| This guide provides general troubleshooting steps and solutions to possible issues that
  may be encountered while installing and configuring OpenVINO™. For a comprehensive 
  database of support topics on OpenVINO, go to:
| `Support for OpenVINO™ toolkit <https://www.intel.com/content/www/us/en/support/products/96066/software/development-software/openvino-toolkit.html>`__



.. _install_for_prc:

.. dropdown:: Errors with Installing via PIP for Users in China

   Users in China might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try one of the following options:
      
   * Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   .. code-block:: sh
      
      pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install specific framework. For example:
      
   .. code-block:: sh
      
      pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
      

   * For C++ developers, if you have installed OpenVINO Runtime via APT, YUM, or the archive file, and then installed OpenVINO Development Tools via PyPI, you may run into issues. To resolve that, install the components in ``requirements.txt`` by using the following command: 
      
   .. code-block:: sh
      
      pip install -r <INSTALL_DIR>/tools/requirements.txt
      
   For APT and YUM users, replace the ``INSTALL_DIR`` with ``/usr/share/openvino``.

<!-- this part was from Docker installation -->

.. dropdown:: Issues with Installing OpenVINO on Linux from Docker

   .. _proxy-issues:

   Proxy Issues
   ++++++++++++

   If you met proxy issues during the installation with Docker, you need set up proxy settings for Docker. See the `Docker guide <https://docs.docker.com/network/proxy/#set-proxy-using-the-cli>`__ for more details.

.. _yocto_install_issues:

.. dropdown:: Issues with Creating a Yocto Image for OpenVINO

   Error while adding "meta-intel" layer
   +++++++++++++++++++++++++++++++++++++

   When using the ``bitbake-layers add-layer meta-intel`` command, the following error might occur:

   .. code-block:: sh
      
      NOTE: Starting bitbake server...
      ERROR: The following required tools (as specified by HOSTTOOLS) appear to be unavailable in PATH, please install them in order to proceed: chrpath diffstat pzstd zstd


   To resolve the issue, install the ``chrpath diffstat zstd`` tools:

   .. code-block:: sh
      
      sudo apt-get install chrpath diffstat zstd

      3. If you run into issues while installing or configuring OpenVINO™, you can try the following methods to do some quick checks first. 

.. dropdown:: Check the versions of OpenVINO Runtime and Development Tools


   * To check the version of OpenVINO Development Tools, use the following command:

   .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: py
               :force:

               from openvino.tools.mo import convert_model
               ov_model = convert_model(version=True)

         .. tab-item:: CLI
            :sync: cli

            .. code-block:: sh

               mo --version


   * To check the version of OpenVINO Runtime, use the following code:
      
   .. code-block:: sh
      
      from openvino.runtime import get_version 
      get_version()

   
.. dropdown:: Check the versions of Python and PIP

   To check your Python version, run ``python -VV`` or ``python --version``. The supported Python versions should be 64-bit and between 3.8 and 3.11. If you are using Python 3.7, you are recommended to upgrade the version to 3.8 or higher.

   If your Python version does not meet the requirements, update Python:

   * For Windows, **do not install Python from a Windows Store** as it can cause issues. You are highly recommended to install Python from `official website <https://www.python.org/>`__ .
   * For Linux and macOS systems, download and install a proper Python version from `official website <https://www.python.org/>`__ . See the `Python Beginners' Guide <https://wiki.python.org/moin/BeginnersGuide/Download>`__ for more information on selecting a version. Note that macOS 10.x comes with python 2.7 installed, which is not supported, so you must install Python from the official website.

   For PIP, make sure that you have installed the latest version. To check and upgrade your PIP version, run the following command:

   .. code-block:: sh
      
      python -m pip install --upgrade pip

<!--## Check the special tips for Anaconda installation-->

<!--add this part in future-->

.. dropdown:: Check if environment variables are set correctly

   - For Python developers, if you previously installed OpenVINO using the archive file, and are now installing OpenVINO using PIP, remove all the PATH settings and the lines with ``setupvars`` from ``.bashrc``. Note that if you installed OpenVINO with PIP in a virtual environment, you don't need to set any environment variables.
   - If you have installed OpenVINO before, you probably have added ``setupvars`` to your ``PATH /.bashrc`` or Windows environment variables. After restarting your environment, you should see similar information as below: 

   .. code-block:: sh
      
      [setupvars.sh] OpenVINO™ environment initialized
      

   - If you don't see the information above, your PATH variables may be configured incorrectly. Check if you have typed the correct <INSTALL_DIR> or you are trying to activate in the correct directory.
   - If you added it to a ``.bashrc`` file, make sure that the command is correctly written and the file is found in the ``~/.bashrc`` folder.

.. dropdown:: Verify that OpenVINO is correctly installed

   * For Python developers, to verify if OpenVINO is correctly installed, use the following command:

   .. code-block:: sh

      python -c "from openvino.runtime import Core; print(Core().available_devices)"
      
   If OpenVINO was successfully installed, you will see a list of available devices.

   * If you installed OpenVINO Runtime using the archive file, you can search "openvino" in Apps & Features on a Windows system, or check your installation directory on Linux to see if OpenVINO is there.

   * If you installed OpenVINO Runtime from APT, use the ``apt list --installed | grep openvino`` command to list the installed OpenVINO packages.

   * If you installed OpenVINO Runtime from YUM, use the ``yum list installed 'openvino*'`` command to list the installed OpenVINO packages.

.. dropdown:: Check if GPU driver is installed

   :doc:`Additional configurations <openvino_docs_install_guides_configurations_header>` may be required in order to use OpenVINO with different hardware such as Intel® GPUs.

   To run inference on an Intel® GPU, make sure that you have installed the correct GPU driver. To check that, see :doc:`additional configurations for GPU <openvino_docs_install_guides_configurations_for_intel_gpu>`.

.. dropdown:: Check firewall and network settings

   Make sure that your firewall and network settings are configured correctly. For example, consider configuring system-wide proxy settings and specifying options for using PIP behind the proxy: 

   .. code-block:: sh

      pip install --proxy http://address:port --trusted-host pypi.org openvino 

   For specific issues, see :ref:`Errors with Installing via PIP for Users in China <install_for_prc>` and :ref:`proxy issues with installing OpenVINO on Linux from Docker <proxy-issues>`. 


@endsphinxdirective