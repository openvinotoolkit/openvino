# Plugin Testing {#plugin_testing}

Inference Engine (IE) tests infrastructure provides a predefined set of functional tests and utilities. They are used to verify a plugin using the Inference Engine public API.
All the tests are written in the [Google Test C++ framework](https://github.com/google/googletest).

Inference Engine Plugin tests are included in the `IE::funcSharedTests` CMake target which is built within the OpenVINO repository
(see [Build Plugin Using CMake](@ref plugin_build) guide). This library contains tests definitions (the tests bodies) which can be parametrized and instantiated in plugins depending on whether a plugin supports a particular feature, specific sets of parameters for test on supported operation set and so on.

Test definitions are split into tests class declaration (see `inference_engine/tests/functional/plugin/shared/include`) and tests class implementation (see `inference_engine/tests/functional/plugin/shared/src`) and include the following scopes of plugin conformance tests:

1. **Behavior tests** (`behavior` sub-folder), which are a separate test group to check that a plugin satisfies basic Inference
Engine concepts: plugin creation, multiple executable networks support, multiple synchronous and asynchronous inference requests support, and so on. See the next section with details how to instantiate the tests definition class with plugin-specific parameters.

2. **Single layer tests** (`single_layer_tests` sub-folder). This groups of tests checks that a particular single layer can be inferenced on a device. An example of test instantiation based on test definition from `IE::funcSharedTests` library:

    - From the declaration of convolution test class we can see that it's a parametrized GoogleTest based class with the `convLayerTestParamsSet` tuple of parameters:

    @snippet single_layer_tests/convolution.hpp test_convolution:definition

    - Based on that, define a set of parameters for `Template` plugin functional test instantiation:

    @snippet single_layer_tests/convolution.cpp test_convolution:declare_parameters

    - Instantiate the test itself using standard GoogleTest macro `INSTANTIATE_TEST_SUITE_P`:

    @snippet single_layer_tests/convolution.cpp test_convolution:instantiate

3. **Sub-graph tests** (`subgraph_tests` sub-folder). This group of tests is designed to tests small patterns or combination of layers. E.g. when a particular topology is being enabled in a plugin e.g. TF ResNet-50, there is no need to add the whole topology to test tests. In opposite way, a particular repetitive subgraph or pattern can be extracted from `ResNet-50` and added to the tests. The instantiation of the sub-graph tests is done in the same way as for single layer tests.
> **Note**, such sub-graphs or patterns for sub-graph tests should be added to `IE::ngraphFunctions` library first (this library is a pre-defined set of small `ngraph::Function`) and re-used in sub-graph tests after.

4. **HETERO tests** (`subgraph_tests` sub-folder) contains tests for `HETERO` scenario (manual or automatic affinities settings, tests for `QueryNetwork`).

5. **Other tests**, which contain tests for other scenarios and has the following types of tests:
    - Tests for execution graph
    - Etc.

To use these tests for your own plugin development, link the `IE::funcSharedTests` library to your test binary and instantiate required test cases with desired parameters values.

> **NOTE**: A plugin may contain its own tests for use cases that are specific to hardware or need to be extensively tested.

To build test binaries together with other build artifacts, use the `make all` command. For details, see
[Build Plugin Using CMake*](@ref plugin_build).

### Tests for plugin-specific ngraph transformations

Please, refer to [Transformation testing](@ref ngraph_transformation) guide.

### How to Extend Inference Engine Plugin Tests

Inference Engine Plugin tests are open for contribution.
Add common test case definitions applicable for all plugins to the `IE::funcSharedTests` target within the DLDT repository. Then, any other plugin supporting corresponding functionality can instantiate the new test.

All Inference Engine per-layer tests check test layers functionality. They are developed using nGraph functions
as input graphs used by tests. In this case, to test a new layer with layer tests, extend
the `IE::ngraphFunctions` library, which is also included in the Inference Engine Developer package, with a new nGraph function
including the corresponding operation.

> **NOTE**: When implementing a new subgraph test, add new single-layer tests for each operation of the subgraph if such test does not exist.
