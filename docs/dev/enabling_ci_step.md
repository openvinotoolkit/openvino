# Enabling tests in OpenVINO CI

Each component contains own documentation for writing and extending tests.
If you want to get more information about it please read the documentation for interested [component](../../src/README.md).

This guide oversees existed OpenVINO CIs.

OpenVINO has two types of public CIs: [Azure](../../.ci/azure) and [Github actions](../../.github/workflows).

 * [Github actions](../../.github/workflows) is used for documentation build and additional checks.
 * [Azure](../../.ci/azure) is used for public build on different platforms. If you need to run tests from new binary files, you can add it to these configuration files.

## See Also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](./index.md)
