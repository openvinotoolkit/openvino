# Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_movidius_setup_guide}

> **NOTE**: These steps are only required for **IEI Mustang-V100-MX8-R10** card. **IEI Mustang-V100-MX8-R11** card doesn't require any additional steps and it's completely configured using the [general guidance](installing-openvino-linux-ivad-vpu.md).      

## See Also

- [Intel® Movidius™ VPUs Programming Guide for use with the Intel® Distribution of OpenVINO™](movidius-programming-guide.md)
- <a class="download" href="<domain_placeholder>/downloads/595850_Intel_Vision_Accelerator_Design_with_Intel_Movidius™_VPUs-HAL Configuration Guide_rev1.3.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs HAL Configuration Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/613514_Intel Vision Accelerator Design with Intel Movidius™ VPUs Workload Distribution_UG_r0.9.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Workload Distribution User Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/613759_Intel Vision Accelerator Design with Intel Movidius™ VPUs Scheduler_UG_r0.9.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Scheduler User Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/Intel Vision Accelerator Design with Intel Movidius™ VPUs Errata.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Errata</a>

The IEI Mustang-V100-MX8 is an OEM version of the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
This guide assumes you have installed the [Mustang-V100-MX8](https://download.ieiworld.com/) and the [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).

Instructions in this guide for configuring your accelerator include:
1.	Installing the required IEI\* BSL reset software
2.	Configuration settings for the `hddldaemon` service

> **NOTE**: This guide does not apply to Uzel\* cards.

## IEI  Reset Software Installation

Using the IEI Mustang-V100-MX8 requires downloading and installing the most current software for your system.

Visit the [IEI Download Center](https://download.ieiworld.com/) for the most current software and documentation.
Search for **Mustang-V100-MX8**.

Download the appropriate software for your system, decompress the downloaded archive, enter the newly created directory, and run the install script:

On **Linux**\*:
-  Run the `install.sh script` with `sudo`, or as `root`.

On **Windows**\*, do one of the following:<br>
-  **GUI**: Double-click `install.bat`
-  **CLI**: Open a console with administrator privileges, cd into the directory, and run `install.bat`.

## Mustang-V100-MX8 Service Configuration

The `hddldaemon` is a system service, a binary executable that is run to manage the computational workload on the board.  It is a required abstraction layer that handles inference, graphics processing, and any type of computation that should be run on the video processing units (VPUs).  Depending on the board configuration, there can be 8 or 16 VPUs.

> **NOTE**: Graphics and other specialized processing may require some custom development.

### Conventions Used in This Document

`<IE>` refers to the following default OpenVINO&trade; Inference Engine directories:
-  **Linux:**	   
 ```
 /opt/intel/openvino/inference_engine
 ```
-  **Windows:**	    
``` 
C:\Program Files (x86)\IntelSWTools\openvino\inference_engine 
```

If you have installed OpenVINO&trade; in a different directory on your system, you will need to enter your unique directory path.

### Configuration File Location

`<IE>\external\hddl\config\hddl_service.config`

### Service Configuration File Settings

Below are some possible configuration options. 

> **NOTE:**  After changing a configuration file, the `hddldaemon` must be restarted. 

### Recommended Settings

`device_snapshot_mode`       
Changes the output of the `hddldaemon` to display a table with individual VPU statistics.

**Default Setting:**          
`"device_snapshot_mode":    "none"`

**Suggested Setting:**           
`"device_snapshot_mode":    "full"`
 
**Supported Settings:**          
  -  `none` (default)
  -  `base`
  -  `full`

`device_snapshot_style`

**Default Setting:**            
`"device_snapshot_style":    "table"`
   
**Recommended Setting:**           
`"device_snapshot_style":    "table"`  

The `table` setting presents labels on the left for each column and is recommended as easier to read.   
The `tape` setting prints the labels in each column.

**Supported Settings:**     
-  `tape`
-  `table` (default)

`user_group	`  
Restricts the service to group members. 

**Recommended	setting depends on your unique system configuration.**     

**Default Setting**	       
`"user_group":    "users"`

The `hddldaemon` may be restricted to a privileged group of users.  The appropriate group will vary according to the local system configuration.
     
**Supported Settings:**
Valid groups on the current system.  The `"users"` group is a default group that exists on Windows and most Linux distributions.


**Optional Recommended Settings:**                

`"device_utilization" : "off"`                      
This setting displays the percent of time each VPU is in use.  It appears in the `table` column when turned on, or if `"device_fps"` is turned on.

`"memory_usage" : "off"`                    
This setting reports the amount of memory being used by each VPU.

`"max_cycle_switchout": 3`                   
Requires the squeeze scheduler.  This setting might speed up performance significantly, depending on the app.  

> **NOTE:** This setting works in conjunction with: `max_task_number_switch_out`.

`"client_fps" : "off"`                          
This setting reports the total FPS for the dispatching hddl_service (which will have one or more clients per app).

`debug_service`                      
`"debug_service": "false"`                  
(default: `"true"`)


## Additional Resources

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)

- [Intel Distribution of OpenVINO Toolkit documentation](https://docs.openvinotoolkit.org)