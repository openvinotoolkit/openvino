# How to validate

Currently we are rewritting existing kernels using jit IR.
So, to validate he efficiency of implementation via jit IR:

1) Always write an extra kernel instead of replacing / updating existing one, so it is always possible to compare
2) Write an extra kernel implementation using intrinsics as a baseline, to compare final assembly between jit IR and intrinsics implementation which went through the whole compilation stack.
3) Use environment variable to switch between legacy, intrinsics and new jit IR kernels 
4) use src/tests/functional/base_func_tests/include/shared_test_classes/base/benchmark.hpp as a wrapper for existing single_layer_tests / subgraph_tests to run syntetic benchmarks
5) use llvm_project for possible improvements
6) use src/plugins/intel_cpu/thirdparty/onednn/doc/performance_considerations/inspecting_jit.md to inspect generated assembly. For intrinsics use usual compilators disassembly /objdump capabilities
