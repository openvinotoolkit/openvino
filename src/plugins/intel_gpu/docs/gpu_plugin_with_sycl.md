# How to build with DPC++ support

1. Install OneAPI base toolkit. Guide: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-0/installation.html
2. Export environment:
    $ source /opt/intel/oneapi/setvars.sh

3. Configure cmake with the following additional options:
    - [Linux] `-DCMAKE_C_COMPILER:FILEPATH=icx -DCMAKE_CXX_COMPILER:FILEPATH=icpx`
      [Windows] `-DCMAKE_C_COMPILER:FILEPATH=icx -DCMAKE_CXX_COMPILER:FILEPATH=icx`
        - For now find_package(IntelSYCL) doesn't work if compiler is not icpx, so we need to update compilers globally for the whole project
    - `-DENABLE_INTEL_CPU=OFF`
        - OneAPI toolkit with OneDNN installed may cause CPU plugin build issue due to weird include files resolver which prefer system onednn intead of
          CPU fork which causes build issue. Alternatively, OneDNN can be removed from OneAPI toolkit installation.
    - [Linux] `-DCMAKE_CXX_FLAGS:STRING=--gcc-install-dir=/lib/gcc/x86_64-linux-gnu/12/ -DCMAKE_C_FLAGS:STRING=--gcc-install-dir=/lib/gcc/x86_64-linux-gnu/12/`
        - This WA is needed if multiple GCC version available in the system
    - `-DENABLE_SYSTEM_OPENCL=OFF`
        - May help to avoid opencl icd/header conflicts as sycl package may have no clhpp headers
    - `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
        - For some reason with latest OneAPI package versions each `make` call causes full project recompilation, so the recommendation is to enable caching

4. cmake --build . --config Release --parallel
