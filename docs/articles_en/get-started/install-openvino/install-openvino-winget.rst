Install OpenVINO™ Runtime with WinGet
======================================

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows operating systems, using WinGet.

.. note::

   Due to the community-driven nature of this distribution channel, the OpenVINO team does not guarantee timely updates aligned with official releases, nor update availability for all OpenVINO versions.

.. note::

   The WinGet distribution:

   * provides OpenVINO Runtime for Windows x64;
   * is delivered as an MSIX package from the WinGet Community Repository;
   * uses versioned WinGet package identifiers, for example ``Intel.OpenVINOToolkit.2026.2.0``;
   * allows different OpenVINO releases to be installed side by side;
   * does not automatically move an existing project to a newer OpenVINO release line.

Before installing OpenVINO, see the :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

.. note::

   WinGet is a Windows-only installation option. If you need OpenVINO Runtime for Linux or macOS, or if you need Python packages, use another installation method such as archive packages, PyPI or other supported package managers.

Installing OpenVINO Runtime with WinGet
#######################################

1. Check that WinGet is available on your system:

   .. code-block:: bat

      winget --version

   If the command is not available, install or repair App Installer. For details, see the `Microsoft WinGet documentation <https://learn.microsoft.com/en-us/windows/package-manager/winget/>`__ and the `Microsoft App Installer troubleshooting guide <https://learn.microsoft.com/en-us/troubleshoot/windows-client/shell-experience/troubleshoot-apps-start-failure-use-windows-package-manager>`__.

2. Update WinGet package sources:

   .. code-block:: bat

      winget source update

   For details, see the `Microsoft documentation for the WinGet source command <https://learn.microsoft.com/en-us/windows/package-manager/winget/source>`__.

3. Search for the OpenVINO package:

   .. code-block:: bat

      winget search --id Intel.OpenVINOToolkit.2026.2.0 -e --source winget

   For details, see the `Microsoft documentation for the WinGet search command <https://learn.microsoft.com/en-us/windows/package-manager/winget/search>`__.

4. Install OpenVINO Runtime:

   .. code-block:: bat

      winget install --id Intel.OpenVINOToolkit.2026.2.0 -e --source winget

   To install an exact WinGet package version, add the ``--version`` option:

   .. code-block:: bat

      winget install --id Intel.OpenVINOToolkit.2026.2.0 --version 2026.2.0.0 -e --source winget

   For details, see the `Microsoft documentation for the WinGet install command <https://learn.microsoft.com/en-us/windows/package-manager/winget/install>`__.

OpenVINO Runtime is now installed using WinGet.

Installing a specific OpenVINO release
++++++++++++++++++++++++++++++++++++++

OpenVINO packages in WinGet use the OpenVINO release number in the WinGet package identifier. This is intentional and allows you to keep different OpenVINO releases installed side by side.

Use the following WinGet package identifier pattern:

.. code-block:: text

   Intel.OpenVINOToolkit.<MAJOR>.<MINOR>.<PATCH>

For example, after the corresponding packages are published in WinGet, you can install a specific OpenVINO release with one of the following commands:

.. code-block:: bat

   winget install --id Intel.OpenVINOToolkit.2026.2.0 -e --source winget
   winget install --id Intel.OpenVINOToolkit.2026.1.0 -e --source winget
   winget install --id Intel.OpenVINOToolkit.2026.0.0 -e --source winget

The WinGet package version may include an additional revision number, for example ``2026.2.0.0``. Use ``--version`` if you need to select the exact WinGet package version.

Understanding WinGet package identifier and MSIX package name
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

OpenVINO uses two different package names in the WinGet distribution flow:

* ``Intel.OpenVINOToolkit.2026.2.0`` is the WinGet package identifier. Use this identifier with WinGet commands such as ``winget install``, ``winget list``, ``winget show``, and ``winget uninstall``.
* ``IntelCorporation.OpenVINOToolkit.2026.2.0`` is the MSIX package name used by Windows after installation. Use this name with PowerShell commands such as ``Get-AppxPackage``.

This difference is expected. WinGet resolves the package by the WinGet package identifier and installs the MSIX package registered in the manifest. After installation, Windows manages the package as an MSIX package under its MSIX package name.

Verifying the installation
++++++++++++++++++++++++++

To check that the package is installed, use the WinGet package identifier:

.. code-block:: bat

   winget list --id Intel.OpenVINOToolkit.2026.2.0 -e

For details, see the `Microsoft documentation for the WinGet list command <https://learn.microsoft.com/en-us/windows/package-manager/winget/list>`__.

To show package metadata available from WinGet, use the WinGet package identifier:

.. code-block:: bat

   winget show --id Intel.OpenVINOToolkit.2026.2.0 -e --source winget

For details, see the `Microsoft documentation for the WinGet show command <https://learn.microsoft.com/en-us/windows/package-manager/winget/show>`__.

Locating the OpenVINO installation
++++++++++++++++++++++++++++++++++

The WinGet package is installed as an MSIX package, so Windows manages the installation location. To find the installation directory, run the following command in PowerShell with the MSIX package name:

.. code-block:: powershell

   (Get-AppxPackage -Name "IntelCorporation.OpenVINOToolkit.2026.2.0").InstallLocation

For details, see the `Microsoft documentation for Get-AppxPackage <https://learn.microsoft.com/en-us/powershell/module/appx/get-appxpackage>`__.

You can use this path as the OpenVINO installation root in your build scripts or local development environment.

For CMake-based C++ applications, you may set ``OpenVINO_DIR`` to the OpenVINO CMake package location. For example:

.. code-block:: powershell

   $openvinoPackage = Get-AppxPackage -Name "IntelCorporation.OpenVINOToolkit.2026.2.0"
   $env:OpenVINO_DIR = Join-Path $openvinoPackage.InstallLocation "runtime\cmake"

Then configure your application with CMake as usual.

Enabling GPU and NPU devices for inference
++++++++++++++++++++++++++++++++++++++++++

The WinGet package installs OpenVINO Runtime, but it does not install all hardware drivers required for every device. Some hardware, including GPU and NPU, may require additional driver installation or operating system updates.

For more details, see the :doc:`Additional Hardware Setup and Troubleshooting <./configurations>` page and the :doc:`System Requirements page <../../../about-openvino/release-notes-openvino/system-requirements>`.

Uninstalling OpenVINO™ Runtime
##############################

Once OpenVINO Runtime is installed via WinGet, you can remove it using the WinGet package identifier:

.. code-block:: bat

   winget uninstall --id Intel.OpenVINOToolkit.2026.2.0 -e --source winget

For details, see the `Microsoft documentation for the WinGet uninstall command <https://learn.microsoft.com/en-us/windows/package-manager/winget/uninstall>`__.

If you have multiple OpenVINO releases installed side by side, run the uninstall command for each WinGet package identifier you want to remove.

What's Next?
############

Now that you have installed OpenVINO Runtime, you are ready to run your own machine learning applications. To learn more about how to integrate a model in OpenVINO applications, try out some tutorials and sample applications.

Try the :doc:`C++ Quick Start Example <../../../get-started/learn-openvino/openvino-samples/get-started-demos>` for step-by-step instructions on building and running a basic image classification C++ application.

Visit the :doc:`Samples <../../../get-started/learn-openvino/openvino-samples>` page for other C++ example applications to get you started with OpenVINO, such as:

* :doc:`Basic object detection with the Hello Reshape SSD C++ sample <../../../get-started/learn-openvino/openvino-samples/hello-reshape-ssd>`
* :doc:`Object classification sample <../../../get-started/learn-openvino/openvino-samples/hello-classification>`
