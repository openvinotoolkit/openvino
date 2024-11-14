Optimizing memory usage
=======================


.. warning::

   Before applying any of the recommendations provided here, note that it may significantly
   impact first-inference latency.

The most RAM-consuming OpenVINO stage is model compilation. It may cause several issues:

* Not enough memory to compile a model. To decrease memory requirement, the following options may be applied:

  * Weights mapping - memory mapping (using ``mmap``) has been introduced as the default way to work
    with weights. Currently, this feature is supported by the IR and ONNX frontends.
    Mapping may be switched by specifying the ``ov::enable_mmap(BOOL)`` property for the ``ov::Core``.
    Because of its "memory-on-demand" nature, there is no need to store all weights
    in RAM. Storing just the data that is needed at the moment lowers the amount of memory
    required for compilation. Moreover, ``mmap`` provides extensive memory sharing, so the
    consecutive compilation of the same model will fetch the information already stored in RAM
    instead of reading it one more time from storage.

  * Decrease the number of threads for compilation - to change the number of threads, specify
    the ``ov::compilation_num_threads(NUMBER)`` property for the ``ov::Core`` or pass it as an additional
    argument to ``ov::Core::compile_model()``

* Not enough memory to recompile a model. If model compilation is successful but one of the
  following recompilations fails due lack of resources, it may be caused by:

  * Memory leak - to determine direct leaks, you can use tools like 'address-sanitizer' or
    'valgrind'. In case of indirect leaks, which cannot be caught by tools, peak RAM (VMHWM)
    may be tracked (you can use tests/stress_tests/memleaks_tests as a tracking tool). If you
    experience significant memory usage increase, report it in
    `Github "Issues" <https://github.com/openvinotoolkit/openvino/issues>`__

  * Memory allocator behavior - each allocator works according to a unique strategy and
    balances between performance and memory usage. For example, the GNU allocator aggressively
    requests from the OS for more memory for consecutive model compilations than was
    required for the first compilation (such behavior may be determined by tracking actual RAM
    (VMRSS) after compilation - it will grow until some stable point). To optimize memory
    pressure, the following options are available:

    * Apply ``malloc_trim(0)``. The function attempts to release free memory even from thread
      caches, so it may signifficantly decrease and stabilize VMRSS usage

    * Use glibc ``Tunables``. A couple of promising options are:
      ``glibc.malloc.trim_threshold`` and `glibc.malloc.arena_max`.
      More details on the two may be found in the
      `GNU Tunables Manual <https://www.gnu.org/software/libc/manual/html_node/Tunables.html>`__

    * Try another allocator. One of the allocators that handles memory carefully is ``jemalloc``

