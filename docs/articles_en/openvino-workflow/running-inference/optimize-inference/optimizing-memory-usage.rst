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

  * Temporary mapping for generated constants - constants allocated by OpenVINO while reading
    a model or running graph transformations can use temporary ``mmap``-backed files. Set
    ``ov::max_memory(BYTES)`` to budget their RAM usage; when the budget is exhausted, further
    constants are backed by files in ``ov::offloading_path(PATH)``. Freed constants return their
    bytes to the budget. Omitting the budget disables offloading; explicitly setting it to zero
    offloads all eligible constants. An empty path uses the system temporary directory; a
    non-empty path requires a budget. These properties can be set on ``ov::Core`` or passed to
    ``ov::Core::read_model()`` or ``ov::Core::compile_model()``. The budget does not cap total
    process RSS: file-backed pages still count towards RSS but can be reclaimed by the OS.
    This mode is supported on Linux, macOS, and Windows. Use a directory with enough disk space,
    not a RAM-backed temporary filesystem. Each offloaded constant needs its own mapping; on
    Linux the temporary files are unlinked immediately and are invisible in directory listings.

    .. tab-set::

       .. tab-item:: C++
          :sync: cpp

          .. code-block:: cpp

             ov::Core core;
             core.set_property({ov::max_memory(16ULL * 1024ULL * 1024ULL * 1024ULL),
                                ov::offloading_path("/path/to/offload")});
             auto model = core.read_model("model.xml");

       .. tab-item:: Python
          :sync: py

          .. code-block:: py

             import openvino as ov
             from openvino import properties as props

             core = ov.Core()
             core.set_property({props.max_memory: 16 * 1024**3,
                                props.offloading_path: "/path/to/offload"})
             model = core.read_model("model.xml")

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
      caches, so it may significantly decrease and stabilize VMRSS usage

    * Use glibc ``Tunables``. A couple of promising options are:
      ``glibc.malloc.trim_threshold`` and `glibc.malloc.arena_max`.
      More details on the two may be found in the
      `GNU Tunables Manual <https://www.gnu.org/software/libc/manual/html_node/Tunables.html>`__

    * Try another allocator. One of the allocators that handles memory carefully is ``jemalloc``

* The memory may not be restored by the system even if resources are released; this may be caused by allocator behavior and fragmentation.

  * On Linux, the default memory allocator is ``glibc``. If memory usage is high, try tuning malloc parameters. See  `GNU libc manual <https://www.gnu.org/software/libc/manual/html_node/Malloc-Tunable-Parameters.html>`__  for details:
  
    * ``MALLOC_MMAP_THRESHOLD_=13107200`` sets the default value as a static threshold. Adjust this value to balance memory recovery and performance. Note that model compile time can be affected.
    * Try a different memory allocator, such as ``jemalloc``, for more careful memory management.
