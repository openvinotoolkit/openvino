# Inference Engine Test Infrastructure

This is OpenVINO Inference Engine testing framework. OpenVINO Inference Engine test system contains:
* **Unit tests**
  This test type is used for detailed testing of each software instance (including internal classes with their methods)
  within the tested modules (Inference Engine and Plugins). There are following rules which are **required** for Unit
  Tests development:
  * All unit tests are separated into different executables per each tested module.
  * Unit test folder for a particular module should replicate `SRC` folder layout of the corresponding tested module to
    allow further developers get better understanding which part of software is already covered by unit tests and where
    to add new tests if needed.
    > **Example**: There are `network_serializer.h` and `network_serializer.cpp` files within the `src` folder of the
    tested Inference Engine module. Then, a new `network_serializer_test.cpp` file should be created within the root of
    the Unit Test folder for this module. This test file should cover all the classes and methods from the original
    files.

    > **Example**: There is the `ie_reshaper.cpp` file within the `src/shape_infer` subfolder of the tested module. In this case,
    a new `shape_infer` subfolder should be created within the root of the Unit Test folder for this module. And a new
    `ie_reshaper_test.cpp` file should be created within this newly created subfolder. This test file should cover all
    the classes and methods from the original file.

  * Each Unit Test should cover only the target classes and methods. If needed, all external interface components should
    be mocked. There are common mock objects provided within the common Unit Test Utilities to stub the general
    Inference Engine API classes.
    > **Example**: There are `cnn_network_impl.hpp` and `cnn_network_impl.cpp` files within the `src` folder of the tested
    module. In this case, a new `cnn_network_impl_test.cpp` file should be created and it should contain tests on
    `CNNNetworkImpl` class only.

  * It is not prohibited to have several test files for the same file from the tested module.
  * It is not prohibited to create a separate test file for specific classes or functions (not for the whole file).

* **Functional tests**
  This test type is used to verify public Inference Engine API. There are following types of functional tests:
  * `inference_engine_tests` are plugin-independent tests. They are used to verify Inference Engine API methods that do not
    involve any plugin runtime. The examples are: `network_reader`, `network_serializer`, and `precision` tests.
  * `plugin_tests` are plugin-dependent tests. These tests require plugin runtime to be executed during testing. For example,
    any tests using `ExecutableNetwork`, `InferRequest` API can only be implemented within this test group.

  > **Example**: Any new test on creating a CNNNetwork object and checking its output info should be included to
  the Inference Engine Functional tests suite. However, any new test containing reading of a network and loading it to a
  specified plugin is always the plugin test.

  There are following rules which are **required** for Functional Tests development:
  * All Functional tests are separated into different executables for the Inference Engine and each plugin.
  * Pre-converted IR files must not be used within the new Functional Tests. Tested models should be generated during
    the tests execution. The main method to generate a required model is building of the required NGraph function and
    creating a CNNNetwork using it. If a required layer is not covered by Ngraph, it is allowed to build IR file using
    `xml_net_builder` utility (refer to the `ir_net.hpp` file). IR XML files hardcoded as strings within the test
    code should not be used.
  * All the plugin test cases are parameterized with (at least) the device name and included to the common
    `funcSharedTests` static library. This library is linked to the Plugin Test binaries. And all the plugin
    developers just add required test instantiations based on the linked test definitions to own test binary. It should
    be done to make all the **shared** test cases always visible and available to instantiate by other plugins.

    > **NOTE**: Any new plugin test case should be added to the common test definitions library
    (`funcSharedTests`) within the OpenVINO repository first. Then, this test case can be instantiated with the
    required parameters inside own plugin's test binary which links this shared tests library.

    > **NOTE**: `funcSharedTests` library is added to the developer package and available for closed source
    development.
  * All the inference engine functional test cases are defined and instantiated within the single test binary. These
    test cases are not implemented as a separate library and not available for instantiations outside this binary.

* **Inference Engine tests utilities**
  The set of utilities which are used by the Inference Engine Functional and Unit tests. Different helper functions,
  blob comparators, OS-specific constants, etc. are implemented within the utilities.
  Internal namespaces (for example, `ov::test::utils::`) must be used to
  separate utilities by domains.

  > **NOTE**: All the utilities libraries are added to the developer package and available for closed source
  development.

## See also

 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer documentation](../../docs/dev/index.md)
