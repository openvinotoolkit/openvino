Troubleshooting Guide for OpenVINO™ Installation & Configuration
================================================================


.. meta::
   :description: A collection of troubleshooting steps and solutions to possible
                 problems that may occur during the installation and configuration
                 of OpenVINO™ on your system.


| This article provides general troubleshooting steps and solutions to possible issues that you
  may face while installing and configuring OpenVINO™. For a comprehensive database of support
  topics on OpenVINO, go to:
| `Support for OpenVINO™ toolkit <https://www.intel.com/content/www/us/en/support/products/96066/software/development-software/openvino-toolkit.html>`__



.. dropdown:: PIP for Users in China gives errors

   Users in China might encounter errors while downloading sources via PIP during OpenVINO™
   installation. To resolve the issues, try adding the download source using the ``-i``
   parameter with the Python ``pip`` command. For example:

   .. code-block:: sh

      pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/

   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install specific framework. For example:

   .. code-block:: sh

      pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/

.. dropdown:: ImportError: cannot import name 'Core' from 'openvino'

   This error may appear on systems lacking C++ components. Since it is almost exclusively a
   Windows case, installing `Microsoft Visual C++ Redistributable [vc_redist.x64] <https://aka.ms/vs/17/release/vc_redist.x64.exe>`__
   package may fix it. For more information on dependencies, check
   :doc:`System Requirements <../about-openvino/release-notes-openvino/system-requirements>` and
   :doc:`Additional Hardware Configurations <./configurations>`

.. dropdown:: Proxy issues installing OpenVINO on Linux from Docker

   If you face proxy issues during installation with Docker, you may need to set up proxy
   settings for it. See the `Docker guide <https://docs.docker.com/network/proxy/#set-proxy-using-the-cli>`__
   for more details.

.. dropdown:: Check the version of OpenVINO Runtime

   To check the version of OpenVINO Runtime, use the following command:

   .. code-block:: sh

      from openvino.runtime import get_version
      get_version()


.. dropdown:: Check the versions of Python and PIP

   To check your Python version, run ``python -VV`` or ``python --version``. The supported
   Python versions are 64-bit, between 3.9 and 3.12. If your Python version does not meet the
   requirements, you need to upgrade:

   * For Windows, **do not install Python from the Windows Store** as it can cause issues.
     It is highly recommended that you install Python from the
     `official website <https://www.python.org/>`__ .
   * For Linux and macOS systems, download and install a proper Python version from the
     `official website <https://www.python.org/>`__. See the
     `Python Beginners' Guide <https://wiki.python.org/moin/BeginnersGuide/Download>`__
     for more information on selecting a version. Note that macOS 10.x comes with python 2.7
     installed, which is not supported, so you must install Python from the official website.

   For PIP, make sure that you have installed the latest version. To check and upgrade your PIP
   version, run the following command:

   .. code-block:: sh

      python -m pip install --upgrade pip


.. dropdown:: Check if environment variables are set correctly

   * For Python developers, if you previously installed OpenVINO using the archive file,
     and are now installing OpenVINO using PIP, remove all the PATH settings and the lines with
     ``setupvars`` from ``.bashrc``. Note that if you installed OpenVINO with PIP in a virtual
     environment, you don't need to set any environment variables.
   * If you have installed OpenVINO before, you probably have added ``setupvars`` to your
     ``PATH /.bashrc`` or Windows environment variables. After restarting your environment,
     you should see an information similar to the following:

     .. code-block:: sh

        [setupvars.sh] OpenVINO™ environment initialized

   * If you don't see the information above, your PATH variables may be configured incorrectly.
     Check if you have typed the correct <INSTALL_DIR> or you are trying to activate in the
     correct directory.
   * If you added it to a ``.bashrc`` file, make sure that the command is correctly written and
     the file is found in the ``~/.bashrc`` folder.

.. dropdown:: Verify that OpenVINO is correctly installed

   * For Python developers, to verify if OpenVINO is correctly installed, use the following
     command:

     .. code-block:: sh

        python -c "from openvino import Core; print(Core().available_devices)"

     If OpenVINO has been successfully installed, you will see a list of available devices.

   * If you install OpenVINO Runtime using the archive file, you can search "openvino" in
     Apps & Features on a Windows system, or check your installation directory on Linux to see
     if OpenVINO is there.

   * If you install OpenVINO Runtime from APT, use the ``apt list --installed | grep openvino``
     command to list the installed OpenVINO packages.

   * If you install OpenVINO Runtime from YUM, use the ``yum list installed 'openvino*'``
     command to list the installed OpenVINO packages.

.. dropdown:: Check if proper drivers are installed

   :doc:`Additional configurations <configurations>` may be
   required in order to use OpenVINO with different hardware, such as Intel® GPU and NPU.
   Make sure that the device you want to use for inference has the required driver installed,
   as described in :doc:`additional configurations for GPU <configurations/configurations-intel-gpu>`.

.. dropdown:: Check firewall and network settings

   Make sure that your firewall and network settings are configured correctly. For example,
   consider configuring system-wide proxy settings and specifying options for using PIP behind
   the proxy:

   .. code-block:: sh

      pip install --proxy http://address:port --trusted-host pypi.org openvino

   For specific issues, see Errors with Installing via PIP for Users in China and Proxy issues
   with installing OpenVINO on Linux from Docker questions above.

.. dropdown:: a Yocto Image error when adding the "meta-intel" layer

   When using the ``bitbake-layers add-layer meta-intel`` command, the following error might
   occur:

   .. code-block:: sh

      NOTE: Starting bitbake server...
      ERROR: The following required tools (as specified by HOSTTOOLS) appear to be unavailable in PATH, please install them in order to proceed: chrpath diffstat pzstd zstd

   To resolve the issue, install the ``chrpath diffstat zstd`` tools:

   .. code-block:: sh

      sudo apt-get install chrpath diffstat zstd
