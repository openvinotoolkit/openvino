# Optimizing memory usage {#openvino_docs_memory_optimization_guide}

Before applying any of provided recomendations, please, note that it may significantly impact first inference latency. 

The most RAM-consuming OpenVINO stage is model compilation. It may cause several issues:

* Not enough memory to compile model. To decrease memory requirement next options may be applied: 
  
  * IR mapping - we introduced memory mapping (using `mmap`) as a default way to operate with IR. Because of "memory on demand" nature, there is no need to store weights fully in RAM, so it decreases memory level required for compilation. Also `mmap` provides extensive memory sharing, so next compilation of same model will fetch same memory from RAM instead of additional read from storage

  * Decrease number of threads for compilation - to change number of threads, specify ``ov::compilation_num_threads(NUMBER)`` property for the ``ov::Core`` or pass as additional argument to ``ov::Core::compile_model()``

* Not enough memory to recompile model. If model compilation was successful, but one of next recompilations failed due lack of resources, it may be caused by:

  * Memory leak - to determine direct leaks tools like `address-sanitizer`, `valgrind` etc. may be used. In case of indirect leaks that can't be catched by tools, peak RAM (VMHWM) may be tracked (as tracking tool tests/stress_tests/memleaks_tests may be used). If you experience significant memory usage increase, please, report to [Github "Issues"](https://github.com/openvinotoolkit/openvino/issues)

  * Memory allocator behavior - each allocator works according to unique strategy and balancing between performance and memory usage. For example, GNU allocator aggressively requests more memory from OS for every next model compilation than was required for very first compilation (such behavior may be determined by tracking actual RAM (VMRSS) after compilation, it will grow until some stable point). To optimize memory pressure, next options are available:

    * Apply `malloc_trim(0)`. The function attempts to release free memory even from threads caches, so it may singificantly decrease and stabilize VMRSS usage

    * Use glibc `Tunables`. A few promising options are `glibc.malloc.trim_threshold` and `glibc.malloc.arena_max`. More details may be found here: https://www.gnu.org/software/libc/manual/html_node/Tunables.html

    * Try another allocator. One of the allocators which carefully operates with memory is `jemalloc`