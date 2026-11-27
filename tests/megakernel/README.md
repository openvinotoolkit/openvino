# GPU megakernel integration test

Configure a separate OpenVINO build with the test backend and Google Test enabled:

```sh
cmake -S . -B build-megakernel-test \
  -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL -DENABLE_TESTS=ON \
  -DOUTPUT_ROOT="$PWD/build-megakernel-test/output" \
  -DMEGAKERNEL_ROOT="$PWD/tests/megakernel" -DMEGAKERNEL_IMPLEMENTATION=megakernel_test_lib
cmake --build build-megakernel-test --target ov_megakernel_integration_test --parallel
ctest --test-dir build-megakernel-test -R '^ov_megakernel_integration_test$' --output-on-failure
```

The test needs an Intel OpenCL GPU with USM support. It compares a fused Add/Relu against the same GPU graph with megakernel disabled and checks that the test runtime executed.

The Ubuntu 24.04 GitHub Actions workflow builds this configuration separately and runs the integration test on an Intel GPU runner when GPU tests are affected.