# How to build with DPC++ support

1. Install OneAPI base toolkit. Guide: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-0/installation.html
2. Export environment:
    $ source /opt/intel/oneapi/setvars.sh

3. configure cmake with the following additional options:
    - `-DCMAKE_CXX_FLAGS:STRING=--gcc-install-dir=/lib/gcc/x86_64-linux-gnu/12/ -DCMAKE_C_FLAGS:STRING=--gcc-install-dir=/lib/gcc/x86_64-linux-gnu/12/`
        - This WA is needed if multiple GCC version available in the system
    - `-DCMAKE_CXX_STANDARD:STRING=17`
        - Sycl requires c++17
    - `-DENABLE_SYSTEM_OPENCL=OFF`
        - May help to avoid opencl icd/header conflicts as sycl package may have no clhpp headers
    - `-DCMAKE_C_COMPILER:FILEPATH=icx -DCMAKE_CXX_COMPILER:FILEPATH=icpx`
        - For now find_package(IntelSYCL) doesn't work if compiler is not icpx
4. make -j$(nproc)
