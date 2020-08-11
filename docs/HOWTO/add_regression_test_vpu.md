# Regression tests howto {#openvino_docs_HOWTO_add_regression_test_vpu}

## Purpose

This document contains instructions for correctly modifying a set of regression tests.

## Common

Regression tests for Myriad and HDDL plugins are on the path:
`inference-engine/tests/functional/vpu/regression_tests/`

The tests are divided into 4 groups:
* Classification
* Detection
* Raw-results
* Compilation
* VPU hetero

Testing  framework – [Google Test](https://github.com/google/googletest/).
Each group contains [parameterized](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md) tests. The main idea is that to add a new test, you only need to add a new parameter. Except for scenarios different from the generalized case.

## Classsification and Detection tests

These groups contains two cases:

* For generalized scenario (` VpuNoClassificationRegression, VpuNoDetectionRegression`)
* For specific scenario (` VpuNoClassificationRegressionSpecific, VpuNoDetectionRegressionSpecific`)

### Generalized scenario

If You want test new parameter(batch, precision, model and etc.) then You need to edit the existing initialization of parameterized tests or create a new one.  
Example of initialization of parameterized tests:

``` c++
INSTANTIATE_TEST_CASE_P(
        VPURegTestWithResources_nightly,
        VpuNoClassificationRegression,
        Combine(ValuesIn(VpuTestParamsContainer::testingPlugin()),
                Values(Precision::FP16),
                Values(1),  // batches
                Values(true), //IsHwAdaptiveMode
                Values(false), //DoReshape
                Values(3, 5, 7), //Resources
                Values(false), //IsIgnoreStatistic
                Values(ClassificationSrcParam{ModelName::GoogleNetV1, SourceImages::kCat3, 0.01, Regression::EMean::eValues})),
        VpuNoClassificationRegression::getTestCaseName);
```

### Specific scenario

If You need a test to perform some actions that are not provided in the generalized scenario, then add a specific test case. As with the generalized scenario You can change parameters for these tests.  
Example of specific test case:

``` c++
TEST_P(VpuNoClassificationRegressionSpecific, onAlexNetWithNetworkConfig) {
    DISABLE_ON_WINDOWS_IF(HDDL_PLUGIN);
    DISABLE_IF(do_reshape_);

    if (!hw_adaptive_mode_) {
        config_[VPU_CONFIG_KEY(NETWORK_CONFIG)] = "data=data,scale=1";
    }

    assertThat().classificationResultsForInferRequestAPI()
            .on(SourceImages::kDog2)
            .withInputPrecision(in_precision_)
            .times(batch_)
            .withBatch(batch_)
            .onModel(ModelName::AlexNet)
            .setMean(Regression::EMean::eImage)
            .onFP16()
            .withTopK(1)
            .withPluginConfig(config_)
            .equalToReferenceWithDelta(0.04);
}
```

## Raw-results tests

There is no generalized scenario and recommendations are the same as for specific test cases for Classification/Detection groups.

## Compilation tests

The tests are in the `vpu_classification_regression.cpp` file and contains only one scenario ` VpuNoRegressionWithCompilation `. To add a new test just update parameters just as in generalized scenarion of Classification/Detection test groups.
