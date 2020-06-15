# Plugin Testing {#plugin_testing}

Inference Engine (IE) tests infrastructure provides a predefined set of functional tests and utilities exported via the Inference
Engine developer package. They are used to verify a plugin using the Inference Engine public API.
All the tests are written in the [Google Test C++ framework](https://github.com/google/googletest).

To build test binaries together with other build artifacts, use the `make all` command. For details, see
[Build Plugin Using CMake*](@ref plugin_build).

Inference Engine Plugin tests are included in the `funcSharedTests` CMake target which is built within the  Deep Learning Deployment Toolkit (DLDT) repository
(see [Build Plugin Using CMake](@ref plugin_build) guide).

Test definitions:

1. **Conformance tests**, which are a separate test group to check that a plugin satisfies basic Inference
Engine concepts: plugin creation, multiple executable networks support, multiple synchronous and asynchronous inference requests support, and so on.
2. **Other API tests**, which contain the following types of tests:
    - Per-layer tests. Located in the `single_layer_tests`and `subgraph_tests` folders.
    - Tests for integration with the `InferenceEngine::Core` class. Located in the the `ie_class` folder.
    - Tests to check that IE common preprocessing works with your plugin. The `io_blob_tests` folder.

To use these tests for your own plugin development, link the `funcSharedTests` library to your test binary and
instantiate required test cases with desired parameters values.

> **NOTE**: A plugin may contain its own tests for use cases that are specific to hardware or need to be extensively
> tested. Depending on your device positioning, you can implement more specific tests for your device. Such tests can
> be defined both for conformance and other API tests groups within your own test binary.

How to Extend Inference Engine Plugin Tests
========================

Inference Engine Plugin tests are open for contribution.
Add common test case definitions applicable for all plugins to the `funcSharedTests` target within the DLDT repository. Then, any other plugin supporting corresponding functionality can instantiate the new test.

All Inference Engine per-layer tests check test layers functionality. They are developed using nGraph functions
as input graphs used by tests. In this case, to test a new layer with layer tests, extend
the `ngraphFunctions` CMake target, which is also included in the Inference Engine Developer package, with a new nGraph function
including the corresponding operation.

> **NOTE**: When implementing a new subgraph test, add new single-layer tests for each operation of the subgraph.