# OpenVINO Core Tests

OpenVINO Core is covered by tests from the binary `ov_core_unit_tests`. This binary can be built by the target with the same name.

OpenVINO Core [tests](../tests/) have next structure:
 * `conditional_compilation` - tests cover conditional compilation feature
 * `frontend` - PaddlePaddle frontend tests (should be removed in the nearest future)
 * `models` - test models
 * `op_eval` - tests cover evaluate methods
 * `pass` - tests covers common transformations
 * `type_prop` - type and shape propagation tests
 * `visitors` - tests covers visitor API for all supported operations

## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
 * [Test coverage measurement](../../../docs/dev/test_coverage.md)
