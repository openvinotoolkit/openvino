# Configurations for IEI Mustang-V100-MX8-R10 Card {#openvino_docs_install_guides_movidius_setup_guide}

@sphinxdirective

.. note:: These steps are only required for **IEI Mustang-V100-MX8-R10** card. **IEI Mustang-V100-MX8-R11** card doesn't require any additional steps and it's completely configured using the :doc:`general guidance <openvino_docs_install_guides_installing_openvino_ivad_vpu>`

The IEI Mustang-V100-MX8 is an OEM version of the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
This guide assumes you have installed the `Mustang-V100-MX8 <https://download.ieiworld.com/>`_ and OpenVINO™ Runtime.

Instructions in this guide for configuring your accelerator include:

1. Installing the required IEI BSL reset software
2. Configuration settings for the ``hddldaemon`` service

.. note:: This guide does not apply to Uzel cards.

@endsphinxdirective


## Installing IEI Reset Software


@sphinxdirective

Using the IEI Mustang-V100-MX8 requires downloading and installing the most current software for your system.

Visit the `IEI Download Center <https://download.ieiworld.com/>`_ for the most current software and documentation.
Search for **Mustang-V100-MX8**.

Download the appropriate software for your system, extract the downloaded archive file, enter the newly created directory, and run the install script:

On **Linux**:

- Run the ``install.sh script`` with ``sudo``, or as ``root``.

On **Windows**, do one of the following:

- **GUI**: Double-click ``install.bat``
- **CLI**: Open a console with administrator privileges, cd into the directory, and run ``install.bat``.

@endsphinxdirective


## Configuring Mustang-V100-MX8 Service

@sphinxdirective

The ``hddldaemon`` is a system service, a binary executable that is run to manage the computational workload on the board. It is a required abstraction layer that handles inference, graphics processing, and any type of computation that should be run on the video processing units (VPUs). Depending on the board configuration, there can be 8 or 16 VPUs.

.. note:: Graphics and other specialized processing may require some custom development.

@endsphinxdirective


### Conventions Used in This Document

@sphinxdirective

``<OV>`` refers to the following default OpenVINO&trade; Runtime directories:

- **Linux:**

.. code-block::

   /opt/intel/openvino_2022/runtime

- **Windows:**

.. code-block::

   C:\Program Files (x86)\IntelSWTools\openvino\runtime


If you have installed OpenVINO&trade; in a different directory on your system, you will need to enter your unique directory path.

@endsphinxdirective


### Configuration File Location

`<OV>\3rdparty\hddl\config\hddl_service.config`

### Service Configuration File Settings

@sphinxdirective

Below are some possible configuration options.

.. note:: After changing a configuration file, the ``hddldaemon`` must be restarted.

@endsphinxdirective


#### Recommended Settings

@sphinxdirective

.. list-table::
   :header-rows: 1

   * - Setting
     - Default value
     - Recommended value
     - Supported values
     - Description
   * - ``device_snapshot_mode``
     - ``"none"``
     - ``"full"``
     - 
         - ``"none"`` (default)
         - ``"base"``
         - ``"full"``
     - Changes the output of the ``hddldaemon`` to display a table with individual VPU statistics.
   * - ``device_snapshot_style``
     - ``"table"``
     - ``"table"``

       The ``table`` setting presents labels on the left for each column and is recommended as easier to read.
     - 
       - ``"table"`` (default)
       - ``"tape"``

       The ``"tape"`` setting prints the labels in each column.

     - Sets the style for device snapshot.
   * - ``user_group``
     - ``"users"``

       The ``hddldaemon`` may be restricted to a privileged group of users.

       The appropriate group will vary according to the local system configuration.
     - Recommended setting depends on your unique system configuration.
     - Valid groups on the current system.

       The ``"users"`` group is a default group that exists on Windows and most Linux distributions.
     - Restricts the service to group members.


@endsphinxdirective


#### Optional Recommended Settings

@sphinxdirective

.. list-table::
   :header-rows: 1

   * - Setting
     - Recommended Value
     - Description
   * - ``device_utilization``
     - ``"off"``
     - This setting displays the percent of time each VPU is in use.  It appears in the ``table`` column when turned on, or if ``device_fps`` is turned on.
   * - ``memory_usage``
     - ``"off"``
     - This setting reports the amount of memory being used by each VPU.
   * - ``max_cycle_switchout``
     - ``"3"``
     - Requires the squeeze scheduler.  This setting might speed up performance significantly, depending on the app.

       Note that this setting works in conjunction with: ``max_task_number_switch_out``.
   * - ``client_fps``
     - ``"off"``
     - This setting reports the total FPS for the dispatching "hddl_service" (which will have one or more clients per app).
   * - ``debug_service``
     - ``"false"``
     - This setting defines whether to display service log information.

       The default value is ``"true"``

@endsphinxdirective


## Additional Resources

@sphinxdirective

- `Intel Distribution of OpenVINO Toolkit home page <https://software.intel.com/en-us/openvino-toolkit>`_
- :doc:`Troubleshooting Guide <openvino_docs_get_started_guide_troubleshooting>`
- `Intel® Vision Accelerator Design with Intel® Movidius™ VPUs HAL Configuration Guide </downloads/595850_Intel_Vision_Accelerator_Design_with_Intel_Movidius_VPUs-HAL%20Configuration%20Guide_rev1.3.pdf>`_
- `Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Workload Distribution User Guide </downloads/613514_Intel%20Vision%20Accelerator%20Design%20with%20Intel%20Movidius™%20VPUs%20Workload%20Distribution_UG_r0.9.pdf>`_
- `Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Scheduler User Guide </downloads/613759_Intel%20Vision%20Accelerator%20Design%20with%20Intel%20Movidius™%20VPUs%20Scheduler_UG_r0.9.pdf>`_
- `Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Errata </downloads/Intel%20Vision%20Accelerator%20Design%20with%20Intel%20Movidius%20VPUs%20Errata.pdf>`_

@endsphinxdirective
