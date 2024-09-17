Plugin Testing
==============


.. meta::
   :description: Use the openvino::funcSharedTests library, which includes
                 a predefined set of functional tests and utilities to verify a plugin.


OpenVINO tests infrastructure provides a predefined set of functional tests and utilities. They are used to verify a plugin using the OpenVINO public API.
All the tests are written in the `Google Test C++ framework <https://github.com/google/googletest>`__.

OpenVINO Plugin tests are included in the ``openvino::funcSharedTests`` CMake target which is built within the OpenVINO repository
(see :doc:`Build Plugin Using CMake <build-plugin-using-cmake>` guide). This library contains tests definitions (the tests bodies) which can be parametrized and instantiated in plugins depending on whether a plugin supports a particular feature, specific sets of parameters for test on supported operation set and so on.

Test definitions are split into tests class declaration (see ``src/tests/functional/plugin/shared/include``) and tests class implementation (see ``src/tests/functional/plugin/shared/src``) and include the following scopes of plugin conformance tests:

1. **Behavior tests** (``behavior`` sub-folder), which are a separate test group to check that a plugin satisfies basic OpenVINO concepts: plugin creation, multiple compiled models support, multiple synchronous and asynchronous inference requests support, and so on. See the next section with details how to instantiate the tests definition class with plugin-specific parameters.

2. **Single layer tests** (``single_layer_tests`` sub-folder). This groups of tests checks that a particular single layer can be inferenced on a device. An example of test instantiation based on test definition from ``openvino::funcSharedTests`` library:

* From the declaration of convolution test class we can see that it's a parametrized GoogleTest based class with the ``convLayerTestParamsSet`` tuple of parameters:

.. doxygensnippet:: src/tests/functional/shared_test_classes/include/shared_test_classes/single_op/convolution.hpp
   :language: cpp
   :fragment: [test_convolution:definition]

* Based on that, define a set of parameters for ``Template`` plugin functional test instantiation:

.. doxygensnippet:: src/plugins/template/tests/functional/shared_tests_instances/single_layer_tests/convolution.cpp
   :language: cpp
   :fragment: [test_convolution:declare_parameters]

* Instantiate the test itself using standard GoogleTest macro ``INSTANTIATE_TEST_SUITE_P``:

.. doxygensnippet:: src/plugins/template/tests/functional/shared_tests_instances/single_layer_tests/convolution.cpp
   :language: cpp
   :fragment: [test_convolution:instantiate]

3. **Sub-graph tests** (``subgraph_tests`` sub-folder). This group of tests is designed to tests small patterns or combination of layers. E.g. when a particular topology is being enabled in a plugin e.g. TF ResNet-50, there is no need to add the whole topology to test tests. In opposite way, a particular repetitive subgraph or pattern can be extracted from ``ResNet-50`` and added to the tests. The instantiation of the sub-graph tests is done in the same way as for single layer tests.

.. note::

   Such sub-graphs or patterns for sub-graph tests should be added to ``openvino::ov_models`` library first (this library is a pre-defined set of small ``ov::Model``) and re-used in sub-graph tests after.

4. **HETERO tests** (``subgraph_tests`` sub-folder) contains tests for ``HETERO`` scenario (manual or automatic affinities settings, tests for ``query_model``).

5. **Other tests**, which contain tests for other scenarios and has the following types of tests:

   * Tests for execution graph
   * Other

To use these tests for your own plugin development, link the ``openvino::funcSharedTests`` library to your test binary and instantiate required test cases with desired parameters values.

.. note::

   A plugin may contain its own tests for use cases that are specific to hardware or need to be extensively tested.

To build test binaries together with other build artifacts, use the ``make all`` command. For details, see :doc:`Build Plugin Using CMake <build-plugin-using-cmake>`.

How to Extend OpenVINO Plugin Tests
+++++++++++++++++++++++++++++++++++

OpenVINO Plugin tests are open for contribution.
Add common test case definitions applicable for all plugins to the ``openvino::funcSharedTests`` target within the OpenVINO repository. Then, any other plugin supporting corresponding functionality can instantiate the new test.

.. note::

   When implementing a new subgraph test, add new single-layer tests for each operation of the subgraph if such test does not exist.


