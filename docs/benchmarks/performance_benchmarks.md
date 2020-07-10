# Get a Deep Learning Model Performance Boost with Intel® Platforms {#openvino_docs_performance_benchmarks}

## Increase Performance for Deep Learning Inference

The [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit) helps accelerate deep learning inference across a variety of Intel® processors and accelerators. Rather than a one-size-fits-all solution, Intel offers a powerful portfolio of scalable hardware and software solutions, powered by the Intel® Distribution of OpenVINO™ toolkit, to meet the various performance, power, and price requirements of any use case. The benchmarks below demonstrate high performance gains on several public neural networks for a streamlined, quick deployment on **Intel® CPU, VPU and FPGA** platforms. Use this data to help you decide which hardware is best for your applications and solutions, or to plan your AI workload on the Intel computing already included in your solutions.

Measuring inference performance involves many variables and is extremely use-case and application dependent. We use the below four parameters for measurements, which are key elements to consider for a successful deep learning inference application:

1. **Throughput** - Measures the number of inferences delivered within a latency threshold. (for example, number of frames per second). When deploying a system with deep learning inference, select the throughput that delivers the best trade-off between latency and power for the price and performance that meets your requirements.
2. **Value** - While throughput is important, what is more critical in edge AI deployments is the performance efficiency or performance-per-cost. Application performance in throughput per dollar of system cost is the best measure of value.
3. **Efficiency** - System power is a key consideration from the edge to the data center. When selecting deep learning solutions, power efficiency (throughput/watt) is a critical factor to consider. Intel designs provide excellent power efficiency for running deep learning workloads.
4. **Total Benefit** (Most applicable for Intel® VPU Platforms) - Combining the factors of value and efficiency can be a good way to compare which hardware yields the best performance per watt and per dollar for your particular use case. 

---

## Intel® Xeon® E-2124G<a name="xeon-e"></a>

![](img/throughput_xeon_e212g.png)
![](img/value_xeon_e212g.png)
![](img/eff_xeon_e212g.png)

---

## Intel® Xeon® Silver 4216R <a name="xeon-silver"></a>

![](img/throughput_xeon_silver.png)
![](img/value_xeon_silver.png)
![](img/eff_xeon_silver.png)

---

## Intel® Xeon® Gold 5218T <a name="xeon-gold"></a>

![](img/throughput_xeon_gold.png)
![](img/value_xeon_gold.png)
![](img/eff_xeon_gold.png)

---

## Intel® Xeon® Platinum 8270 <a name="xeon-platinum"></a>

![](img/throughput_xeon_platinum.png)
![](img/value_xeon_platinum.png)
![](img/eff_xeon_platinum.png)

---

## Intel® Atom™ x5-E3940 <a name="atom"></a>

![](img/throughput_atom.png)
![](img/value_atom.png)
![](img/eff_atom.png)

---

## Intel® Core™ i3-8100 <a name="core-i3"></a>

![](img/throughput_i3.png)
![](img/value_i3.png)
![](img/eff_i3.png)

---

## Intel® Core™ i5-8500 <a name="core-i5"></a>

![](img/throughput_i5.png)
![](img/value_i5.png)
![](img/eff_i5.png)

---

## Intel® Core™ i7-8700T <a name="core-i7"></a>

![](img/throughput_i7.png)
![](img/value_i7.png)
![](img/eff_i7.png)

---

## Intel® Core™ i9-10920X <a name="core-i9"></a>

![](img/throughput_i9.png)
![](img/value_i9.png)
![](img/eff_i9.png)

---

## Intel® Neural Compute Stick 2 <a name="intel-ncs2"></a>

![](img/throughput_ncs2.png)
![](img/value_ncs2.png)
![](img/eff_ncs2.png)
![](img/benefit_ncs2.png)

---

## Intel® Vision Accelerator Design with Intel® Movidius™ VPUs (Uzel* UI-AR8) <a name="ivad-vpu"></a>

![](img/throughput_hddlr.png)
![](img/value_hddlr.png)
![](img/eff_hddlr.png)

---

## Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA

![](img/throughput_ivad_fpga.png)
![](img/value_ivad_fpga.png)
![](img/eff_ivad_fpga.png)

## Platform Configurations

Intel® Distribution of OpenVINO™ toolkit performance benchmark numbers are based on release 2020.4. 

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Learn more at intel.com, or from the OEM or retailer. Performance results are based on testing as of July 8, 2020 and may not reflect all publicly available security updates. See configuration disclosure for details. No product can be absolutely secure. 

Software and workloads used in performance tests may have been optimized for performance only on Intel microprocessors. Performance tests, such as SYSmark and MobileMark, are measured using specific computer systems, components, software, operations and functions. Any change to any of those factors may cause the results to vary. You should consult other information and performance tests to assist you in fully evaluating your contemplated purchases, including the performance of that product when combined with other products. For more complete information, see [Performance Benchmark Test Disclosure](https://www.intel.com/content/www/us/en/benchmarks/benchmark.html).

Your costs and results may vary. 

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

Optimization Notice: Intel’s compilers may or may not optimize to the same degree for non-Intel microprocessors for optimizations that are not unique to Intel microprocessors. These optimizations include SSE2, SSE3, and SSSE3 instruction sets and other optimizations. Intel does not guarantee the availability, functionality, or effectiveness of any optimization on microprocessors not manufactured by Intel. Microprocessor-dependent optimizations in this product are intended for use with Intel microprocessors. Certain optimizations not specific to Intel microarchitecture are reserved for Intel microprocessors. Please refer to the applicable product User and Reference Guides for more information regarding the specific instruction sets covered by this notice. [Notice Revision #2010804](https://software.intel.com/articles/optimization-notice).

Testing by Intel done on: see test date for each HW platform below.

**CPU Inference Engines**

|                                 | Intel® Xeon® E-2124G  | Intel® Xeon® Silver 4216R    | Intel® Xeon® Gold 5218T      | Intel® Xeon® Platinum 8270   | 
| ------------------------------- | ----------------------| ---------------------------- | ---------------------------- | ---------------------------- |
| Motherboard                     | ASUS* WS C246 PRO     | Intel® Server Board S2600STB | Intel® Server Board S2600STB | Intel® Server Board S2600STB |
| CPU                             | Intel® Xeon® E-2124G CPU @ 3.40GHz | Intel® Xeon® Silver 4216R CPU @ 2.20GHz | Intel® Xeon® Gold 5218T CPU @ 2.10GHz | Intel® Xeon® Platinum 8270 CPU @ 2.70GHz |
| Hyper Threading                 | OFF                   | ON                           | ON                           | ON                           |
| Turbo Setting                   | ON                    | ON                           | ON                           | ON                           |
| Memory                          | 2 x 16 GB DDR4 2666MHz| 12 x 32 GB DDR4 2666MHz      | 12 x 32 GB DDR4 2666MHz      | 12 x 32 GB DDR4 2933MHz      |
| Operating System                | Ubuntu* 18.04 LTS     | Ubuntu* 18.04 LTS            | Ubuntu* 18.04 LTS            | Ubuntu* 18.04 LTS            |
| Kernel Version                  | 5.3.0-24-generic      | 5.3.0-24-generic             | 5.3.0-24-generic             | 5.3.0-24-generic             |
| BIOS Vendor                     | American Megatrends Inc.* | Intel Corporation        | Intel Corporation            | Intel Corporation            |
| BIOS Version                    | 0904                  | SE5C620.86B.02.01.<br>0009.092820190230 | SE5C620.86B.02.01.<br>0009.092820190230 | SE5C620.86B.02.01.<br>0009.092820190230    |
| BIOS Release                    | April 12, 2019        | September 28, 2019           | September 28, 2019           | September 28, 2019           |
| BIOS Settings        | Select optimized default settings, <br>save & exit | Select optimized default settings, <br>change power policy <br>to "performance", <br>save & exit | Select optimized default settings, <br>change power policy to "performance", <br>save & exit | Select optimized default settings, <br>change power policy to "performance", <br>save & exit |
| Batch size                      | 1                     | 1                            | 1                            | 1                            |
| Precision                       | INT8                  | INT8                         | INT8                         | INT8                         |
| Number of concurrent inference requests | 4             | 32                           | 32                           | 52                           |
| Test Date                       |  July 8, 2020        |  July 8, 2020               |  July 8, 2020               |  July 8, 2020               |
| Power dissipation, TDP in Watt  | [71](https://ark.intel.com/content/www/us/en/ark/products/134854/intel-xeon-e-2124g-processor-8m-cache-up-to-4-50-ghz.html#tab-blade-1-0-1)                    | [125](https://ark.intel.com/content/www/us/en/ark/products/193394/intel-xeon-silver-4216-processor-22m-cache-2-10-ghz.html#tab-blade-1-0-1)                          | [105](https://ark.intel.com/content/www/us/en/ark/products/193953/intel-xeon-gold-5218t-processor-22m-cache-2-10-ghz.html#tab-blade-1-0-1)             | [205](https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html#tab-blade-1-0-1)                          |
| CPU Price on July 8, 2020, USD<br>Prices may vary  | [213](https://ark.intel.com/content/www/us/en/ark/products/134854/intel-xeon-e-2124g-processor-8m-cache-up-to-4-50-ghz.html)     | [1,002](https://ark.intel.com/content/www/us/en/ark/products/193394/intel-xeon-silver-4216-processor-22m-cache-2-10-ghz.html)                 | [1,349](https://ark.intel.com/content/www/us/en/ark/products/193953/intel-xeon-gold-5218t-processor-22m-cache-2-10-ghz.html)                        | [7,405](https://ark.intel.com/content/www/us/en/ark/products/192482/intel-xeon-platinum-8270-processor-35-75m-cache-2-70-ghz.html)                        |

**CPU Inference Engines (continue)**

|                      | Intel® Core™ i5-8500               | Intel® Core™ i7-8700T               | Intel® Core™ i9-10920X               |
| -------------------- | ---------------------------------- | ----------------------------------- |--------------------------------------|
| Motherboard          | ASUS* PRIME Z370-A                 | GIGABYTE* Z370M DS3H-CF             | ASUS* PRIME X299-A II                |
| CPU                  | Intel® Core™ i5-8500 CPU @ 3.00GHz | Intel® Core™ i7-8700T CPU @ 2.40GHz | Intel® Core™ i9-10920X CPU @ 3.50GHz |
| Hyper Threading      | OFF                                | ON                                  | ON                                   |
| Turbo Setting        | ON                                 | ON                                  | ON                                   |
| Memory               | 2 x 16 GB DDR4 2666MHz             | 4 x 16 GB DDR4 2400MHz              | 4 x 16 GB DDR4 2666MHz               |
| Operating System     | Ubuntu* 18.04 LTS                  | Ubuntu* 18.04 LTS                   | Ubuntu* 18.04 LTS                    |
| Kernel Version       | 5.3.0-24-generic                   | 5.0.0-23-generic                    | 5.0.0-23-generic                     |
| BIOS Vendor          | American Megatrends Inc.*          | American Megatrends Inc.*           | American Megatrends Inc.*            |
| BIOS Version         | 2401                               | F11                                 | 505                                  |
| BIOS Release         | July 12, 2019                      | March 13, 2019                      | December 17, 2019                    |
| BIOS Settings        | Select optimized default settings, <br>save & exit | Select optimized default settings, <br>set OS type to "other", <br>save & exit | Default Settings |
| Batch size           | 1                                  | 1                                   | 1                                    |
| Precision            | INT8                               | INT8                                | INT8                                 |
| Number of concurrent inference requests | 3               | 4                                   | 24                                   |
| Test Date            | July 8, 2020                       | July 8, 2020                       | July 8, 2020                        |
| Power dissipation, TDP in Watt | [65](https://ark.intel.com/content/www/us/en/ark/products/129939/intel-core-i5-8500-processor-9m-cache-up-to-4-10-ghz.html#tab-blade-1-0-1)                                 | [35](https://ark.intel.com/content/www/us/en/ark/products/129948/intel-core-i7-8700t-processor-12m-cache-up-to-4-00-ghz.html#tab-blade-1-0-1) | [165](https://ark.intel.com/content/www/us/en/ark/products/198012/intel-core-i9-10920x-x-series-processor-19-25m-cache-3-50-ghz.html) |
| CPU Price on  July 8, 2020, USD<br>Prices may vary  | [192](https://ark.intel.com/content/www/us/en/ark/products/129939/intel-core-i5-8500-processor-9m-cache-up-to-4-10-ghz.html)                               | [303](https://ark.intel.com/content/www/us/en/ark/products/129948/intel-core-i7-8700t-processor-12m-cache-up-to-4-00-ghz.html)                                 | [700](https://ark.intel.com/content/www/us/en/ark/products/198012/intel-core-i9-10920x-x-series-processor-19-25m-cache-3-50-ghz.html)

**CPU Inference Engines (continue)**

|                      | Intel Atom® x5-E3940                  | Intel® Core™ i3-8100               | 
| -------------------- | ----------------------------------    |----------------------------------- |
| Motherboard          |                                       | GIGABYTE* Z390 UD                  |
| CPU                  | Intel Atom® Processor E3940 @ 1.60GHz | Intel® Core™ i3-8100 CPU @ 3.60GHz |
| Hyper Threading      | OFF                                   | OFF                                |
| Turbo Setting        | ON                                    | OFF                                |
| Memory               | 1 x 8 GB DDR3 1600MHz                 | 4 x 8 GB DDR4 2400MHz              |
| Operating System     | Ubuntu* 18.04 LTS                     | Ubuntu* 18.04 LTS                  |
| Kernel Version       | 5.3.0-24-generic                      | 5.3.0-24-generic                   |
| BIOS Vendor          | American Megatrends Inc.*             | American Megatrends Inc.*          |
| BIOS Version         | 5.12                                  | F8                                 |
| BIOS Release         | September 6, 2017                     | May 24, 2019                       |
| BIOS Settings        | Default settings                      | Select optimized default settings, <br> set OS type to "other", <br>save & exit |
| Batch size           | 1                                     | 1                                  |
| Precision            | INT8                                  | INT8                               |
| Number of concurrent inference requests | 4                  | 4                                  |
| Test Date            | July 8, 2020                         | July 8, 2020                      |
| Power dissipation, TDP in Watt | [9.5](https://ark.intel.com/content/www/us/en/ark/products/96485/intel-atom-x5-e3940-processor-2m-cache-up-to-1-80-ghz.html)                                                              | [65](https://ark.intel.com/content/www/us/en/ark/products/126688/intel-core-i3-8100-processor-6m-cache-3-60-ghz.html#tab-blade-1-0-1)|
| CPU Price on  July 8, 2020, USD<br>Prices may vary  | [34](https://ark.intel.com/content/www/us/en/ark/products/96485/intel-atom-x5-e3940-processor-2m-cache-up-to-1-80-ghz.html)                                                        | [117](https://ark.intel.com/content/www/us/en/ark/products/126688/intel-core-i3-8100-processor-6m-cache-3-60-ghz.html)       |



**Accelerator Inference Engines**

|              | Intel® Neural Compute Stick 2 | Intel® Vision Accelerator Design<br>with Intel® Movidius™ VPUs (Uzel* UI-AR8) | Intel® Vision Accelerator Design<br>with Intel® Arria® 10 FPGA - IEI/SAF3*| 
| --------------------                    | ------------------------------------- | ------------------------------------- | ------------------------- |
| VPU                                     | 1 X Intel® Movidius™ Myriad™ X MA2485 | 8 X Intel® Movidius™ Myriad™ X MA2485 | 1 X Intel® Arria® 10 FPGA |
| Connection                              | USB 2.0/3.0                           | PCIe X4                               | PCIe X8                   |
| Batch size                              | 1                                     | 1                                     | 1                         |
| Precision                               | FP16                                  | FP16                                  | FP11                      |
| Number of concurrent inference requests | 4                                     | 32                                    | 5                         |
| Power dissipation, TDP in Watt          | 2.5                                   | [30](https://www.mouser.com/ProductDetail/IEI/MUSTANG-V100-MX8-R10?qs=u16ybLDytRaZtiUUvsd36w%3D%3D)                                    | [60](https://www.mouser.com/ProductDetail/IEI/MUSTANG-F100-A10-R10?qs=sGAEpiMZZMtNlGR3Dbecs5Qs0RmP5oxxCbTJPjyRuMXthliRUwiVGw%3D%3D) |
| CPU Price, USD<br>Prices may vary | [69](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) (from July 8, 2020)                                | [768](https://www.mouser.com/ProductDetail/IEI/MUSTANG-V100-MX8-R10?qs=u16ybLDytRaZtiUUvsd36w%3D%3D) (from May 15, 2020)               | [1,650](https://www.bhphotovideo.com/c/product/1477989-REG/qnap_mustang_f100_a10_r10_pcie_fpga_highest_performance.html/?ap=y&ap=y&smp=y&msclkid=371b373256dd1a52beb969ecf5981bf8) (from July 8, 2020)                    |
| Host Computer                           | Intel® Core™ i7                       | Intel® Core™ i5                       | Intel® Core™ i5           |
| Motherboard                             | ASUS* Z370-A II                       | Uzelinfo* / US-E1300                  | IEI/SAF3*              |
| CPU                                     | Intel® Core™ i7-8700 CPU @ 3.20GHz    | Intel® Core™ i5-6600 CPU @ 3.30GHz    | Intel® Core™ i5-7500T CPU @ 2.70GHz |
| Hyper Threading                         | ON                                    | OFF                                   | OFF                       |
| Turbo Setting                           | ON                                    | ON                                    | ON                        |
| Memory                                  | 4 x 16 GB DDR4 2666MHz                | 2 x 16 GB DDR4 2400MHz                | 2 x 16 GB DDR4 2666MHz    |
| Operating System                        | Ubuntu* 18.04 LTS                     | Ubuntu* 18.04 LTS                     | Ubuntu* 16.04 LTS         |
| Kernel Version                          | 5.0.0-23-generic                      | 5.0.0-23-generic                      | 4.13.0-45-generic         |
| BIOS Vendor                             | American Megatrends Inc.*             | American Megatrends Inc.*             | American Megatrends Inc.* |
| BIOS Version                            | 411                                   | 5.12                                  | V2RMAR15                  |
| BIOS Release                            | September 21, 2018                    | September 21, 2018                    | December 03, 2019         |
| Test Date                               | July 8, 2020                         | July 8, 2020                         | July 8, 2020          |

Please follow this link for more detailed configuration descriptions: [Configuration Details](https://docs.openvinotoolkit.org/resources/benchmark_files/system_configurations_2020.4.html)

\htmlonly
<style>
    .footer {
        display: none;
    }
</style>
<div class="opt-notice-wrapper">
<p class="opt-notice">
\endhtmlonly
For more complete information about performance and benchmark results, visit: [www.intel.com/benchmarks](https://www.intel.com/benchmarks) and [Optimization Notice](https://software.intel.com/articles/optimization-notice). [Legal Information](../Legal_Information.md).
\htmlonly
</p>
</div>
\endhtmlonly