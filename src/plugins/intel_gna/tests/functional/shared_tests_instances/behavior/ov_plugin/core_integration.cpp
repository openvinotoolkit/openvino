// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/ov_plugin/core_integration.hpp"

#include "behavior/compiled_model/properties.hpp"
#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "gna/gna_config.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

using namespace ov::test::behavior;

namespace {

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(nightly_OVBasicPropertiesTestsP,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_intel_gna_plugin", "GNA")));

// TODO
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassModelTestP, OVClassModelTestP, ::testing::Values("GNA"));
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassModelOptionalTestP, OVClassModelOptionalTestP, ::testing::Values("GNA"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_MultiHeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("MULTI", "HETERO"));

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(
    smoke_MultiHeteroOVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI", "HETERO"),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_GNA = {
    {"GNA", std::make_pair(ov::AnyMap{}, "GNA")}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_GNA));

INSTANTIATE_TEST_SUITE_P(nightly_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values("GNA"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVPropertiesDefaultSupportedTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values("GNA"));

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedScaleFactors) {
    ov::Core core;
    float sf1, sf2;
    OV_ASSERT_NO_THROW(core.set_property({{"GNA_SCALE_FACTOR_0", "1634.0"}, {"GNA_SCALE_FACTOR_1", "2000.0"}}));
    OV_ASSERT_NO_THROW(sf1 = std::stof(core.get_property("GNA", "GNA_SCALE_FACTOR_0").as<std::string>()));
    OV_ASSERT_NO_THROW(sf2 = std::stof(core.get_property("GNA", "GNA_SCALE_FACTOR_1").as<std::string>()));
    ASSERT_FLOAT_EQ(1634.0, sf1);
    ASSERT_FLOAT_EQ(2000.0, sf2);

    ASSERT_THROW(core.set_property("GNA",
                                   {ov::intel_gna::scale_factors_per_input(
                                        std::map<std::string, float>{{"input_0", 1634.0f}, {"input_1", 2000.0f}}),
                                    {"GNA_SCALE_FACTOR_0", "1634.0"},
                                    {"GNA_SCALE_FACTOR_1", "2000.0"}}),
                 ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedScaleFactorsPerInput) {
    ov::Core core;
    std::map<std::string, float> scale_factors_per_input;

    OV_ASSERT_NO_THROW(
        core.set_property("GNA",
                          ov::intel_gna::scale_factors_per_input(
                              std::map<std::string, float>{{"input_0", 1634.0f}, {"input_1", 2000.0f}})));
    OV_ASSERT_NO_THROW(scale_factors_per_input = core.get_property("GNA", ov::intel_gna::scale_factors_per_input));
    ASSERT_EQ(2, scale_factors_per_input.size());
    ASSERT_FLOAT_EQ(1634.0f, scale_factors_per_input["input_0"]);
    ASSERT_FLOAT_EQ(2000.0f, scale_factors_per_input["input_1"]);

    OV_ASSERT_NO_THROW(
        core.set_property("GNA", ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}})));
    OV_ASSERT_NO_THROW(scale_factors_per_input = core.get_property("GNA", ov::intel_gna::scale_factors_per_input));
    ASSERT_EQ(1, scale_factors_per_input.size());
    ASSERT_FLOAT_EQ(1.0f, scale_factors_per_input["0"]);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedPrecisionHint) {
    ov::Core core;
    ov::element::Type precision;

    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    ASSERT_EQ(ov::element::undefined, precision);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::inference_precision(ov::element::i8)));
    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    ASSERT_EQ(ov::element::i8, precision);

    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::inference_precision(ov::element::i8)));
    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    OPENVINO_SUPPRESS_DEPRECATED_END

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::inference_precision(ov::element::i16)));
    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    ASSERT_EQ(ov::element::i16, precision);

    OV_ASSERT_NO_THROW(core.set_property("GNA", {{ov::hint::inference_precision.name(), "I8"}}));
    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    ASSERT_EQ(ov::element::i8, precision);

    OV_ASSERT_NO_THROW(core.set_property("GNA", {{ov::hint::inference_precision.name(), "I16"}}));
    OV_ASSERT_NO_THROW(precision = core.get_property("GNA", ov::hint::inference_precision));
    ASSERT_EQ(ov::element::i16, precision);

    OV_ASSERT_NO_THROW(
        core.set_property("GNA", {ov::hint::inference_precision(ov::element::i8), {GNA_CONFIG_KEY(PRECISION), "I16"}}));
    ASSERT_THROW(core.set_property("GNA", ov::hint::inference_precision(ov::element::i32)), ov::Exception);
    ASSERT_THROW(core.set_property("GNA", ov::hint::inference_precision(ov::element::undefined)), ov::Exception);
    ASSERT_THROW(core.set_property("GNA", {{ov::hint::inference_precision.name(), "ABC"}}), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedPerformanceHint) {
    ov::Core core;
    ov::hint::PerformanceMode mode;

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));
    OV_ASSERT_NO_THROW(mode = core.get_property("GNA", ov::hint::performance_mode));
    ASSERT_EQ(ov::hint::PerformanceMode::LATENCY, mode);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)));
    OV_ASSERT_NO_THROW(mode = core.get_property("GNA", ov::hint::performance_mode));
    ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT, mode);

    ASSERT_THROW(core.set_property("GNA", {{ov::hint::performance_mode.name(), "ABC"}}), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedNumRequests) {
    ov::Core core;
    uint32_t num_requests;

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::num_requests(8)));
    OV_ASSERT_NO_THROW(num_requests = core.get_property("GNA", ov::hint::num_requests));
    ASSERT_EQ(8, num_requests);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::num_requests(1)));
    OV_ASSERT_NO_THROW(num_requests = core.get_property("GNA", ov::hint::num_requests));
    ASSERT_EQ(1, num_requests);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::num_requests(1000)));
    OV_ASSERT_NO_THROW(num_requests = core.get_property("GNA", ov::hint::num_requests));
    ASSERT_EQ(127, num_requests);  // maximum value

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::hint::num_requests(0)));
    OV_ASSERT_NO_THROW(num_requests = core.get_property("GNA", ov::hint::num_requests));
    ASSERT_EQ(1, num_requests);  // minimum value

    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_ASSERT_NO_THROW(core.set_property("GNA", {ov::hint::num_requests(8), {GNA_CONFIG_KEY(LIB_N_THREADS), "8"}}));
    ASSERT_THROW(core.set_property("GNA", {ov::hint::num_requests(4), {GNA_CONFIG_KEY(LIB_N_THREADS), "8"}}),
                 ov::Exception);
    OPENVINO_SUPPRESS_DEPRECATED_END
    ASSERT_THROW(core.set_property("GNA", {{ov::hint::num_requests.name(), "ABC"}}), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedExecutionMode) {
    ov::Core core;
    auto execution_mode = ov::intel_gna::ExecutionMode::AUTO;

    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::SW_EXACT, execution_mode);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32)));
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::SW_FP32, execution_mode);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT)));
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::SW_EXACT, execution_mode);

    OV_ASSERT_NO_THROW(
        core.set_property("GNA", ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK)));
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK, execution_mode);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW)));
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::HW, execution_mode);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::AUTO)));
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::AUTO, execution_mode);

    ASSERT_THROW(core.set_property("GNA", {{ov::intel_gna::execution_mode.name(), "ABC"}}), ov::Exception);
    OV_ASSERT_NO_THROW(execution_mode = core.get_property("GNA", ov::intel_gna::execution_mode));
    ASSERT_EQ(ov::intel_gna::ExecutionMode::AUTO, execution_mode);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedTargetDevice) {
    ov::Core core;
    auto execution_target = ov::intel_gna::HWGeneration::UNDEFINED;
    auto compile_target = ov::intel_gna::HWGeneration::UNDEFINED;

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_2_0)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_2_0, execution_target);
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::compile_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_2_0, compile_target);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_0)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_0, execution_target);
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::compile_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_2_0, compile_target);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_5)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_5, execution_target);
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::compile_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_2_0, compile_target);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_3_0)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_5, execution_target);
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::compile_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_0, compile_target);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_3_5)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_5, execution_target);
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::compile_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::GNA_3_5, compile_target);

    OV_ASSERT_NO_THROW(
        core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::UNDEFINED)));
    OV_ASSERT_NO_THROW(execution_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::UNDEFINED, execution_target);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::UNDEFINED)));
    OV_ASSERT_NO_THROW(compile_target = core.get_property("GNA", ov::intel_gna::execution_target));
    ASSERT_EQ(ov::intel_gna::HWGeneration::UNDEFINED, compile_target);

    ASSERT_THROW(core.set_property("GNA",
                                   {ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_2_0),
                                    {GNA_CONFIG_KEY(EXEC_TARGET), "GNA_TARGET_3_0"}}),
                 ov::Exception);
    ASSERT_THROW(core.set_property("GNA",
                                   {ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_2_0),
                                    {GNA_CONFIG_KEY(COMPILE_TARGET), "GNA_TARGET_3_0"}}),
                 ov::Exception);

    ASSERT_THROW(core.set_property("GNA",
                                   {ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_2_0),
                                    {GNA_CONFIG_KEY(EXEC_TARGET), "GNA_TARGET_3_5"}}),
                 ov::Exception);
    ASSERT_THROW(core.set_property("GNA",
                                   {ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_2_0),
                                    {GNA_CONFIG_KEY(COMPILE_TARGET), "GNA_TARGET_3_5"}}),
                 ov::Exception);

    ASSERT_THROW(core.set_property("GNA", {{ov::intel_gna::execution_target.name(), "ABC"}}), ov::Exception);
    ASSERT_THROW(core.set_property("GNA", {{ov::intel_gna::compile_target.name(), "ABC"}}), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedPwlAlgorithm) {
    ov::Core core;
    auto pwl_algo = ov::intel_gna::PWLDesignAlgorithm::UNDEFINED;
    float pwl_max_error = 0.0f;

    OV_ASSERT_NO_THROW(
        core.set_property("GNA",
                          ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::RECURSIVE_DESCENT)));
    OV_ASSERT_NO_THROW(pwl_algo = core.get_property("GNA", ov::intel_gna::pwl_design_algorithm));
    ASSERT_EQ(ov::intel_gna::PWLDesignAlgorithm::RECURSIVE_DESCENT, pwl_algo);

    OV_ASSERT_NO_THROW(core.set_property(
        "GNA",
        ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION)));
    OV_ASSERT_NO_THROW(pwl_algo = core.get_property("GNA", ov::intel_gna::pwl_design_algorithm));
    ASSERT_EQ(ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION, pwl_algo);

    ASSERT_THROW(core.set_property("GNA", {{ov::intel_gna::pwl_design_algorithm.name(), "ABC"}}), ov::Exception);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::pwl_max_error_percent(0.05)));
    OV_ASSERT_NO_THROW(pwl_max_error = core.get_property("GNA", ov::intel_gna::pwl_max_error_percent));
    ASSERT_FLOAT_EQ(0.05, pwl_max_error);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::pwl_max_error_percent(100.0f)));
    OV_ASSERT_NO_THROW(pwl_max_error = core.get_property("GNA", ov::intel_gna::pwl_max_error_percent));
    ASSERT_FLOAT_EQ(100.0f, pwl_max_error);

    OPENVINO_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(
        core.set_property("GNA",
                          {ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::RECURSIVE_DESCENT),
                           {GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN), InferenceEngine::PluginConfigParams::YES}}),
        ov::Exception);
    OPENVINO_SUPPRESS_DEPRECATED_END
    ASSERT_THROW(core.set_property("GNA", ov::intel_gna::pwl_max_error_percent(-1.0f)), ov::Exception);
    ASSERT_THROW(core.set_property("GNA", ov::intel_gna::pwl_max_error_percent(146.0f)), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedLogLevel) {
    ov::Core core;
    auto level = ov::log::Level::NO;

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::INFO)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::INFO, level);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::ERR)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::ERR, level);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::WARNING)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::WARNING, level);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::DEBUG)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::DEBUG, level);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::TRACE)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::TRACE, level);

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::log::level(ov::log::Level::NO)));
    OV_ASSERT_NO_THROW(level = core.get_property("GNA", ov::log::level));
    ASSERT_EQ(ov::log::Level::NO, level);

    ASSERT_THROW(core.set_property("GNA", {{ov::log::level.name(), "NO"}}), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedFwModelPathNegative) {
    ov::Core core;
    std::string path = "";

    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_5)));
    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::firmware_model_image_path("model.bin")));
    ASSERT_THROW(path = core.get_property("GNA", ov::intel_gna::firmware_model_image_path), ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAfterCreatedFwModelPathPositive) {
    ov::Core core;
    std::string path = "";

    OV_ASSERT_NO_THROW(
        core.set_property("GNA", ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_5_E)));
    OV_ASSERT_NO_THROW(core.set_property("GNA", ov::intel_gna::firmware_model_image_path("model.bin")));
    OV_ASSERT_NO_THROW(path = core.get_property("GNA", ov::intel_gna::firmware_model_image_path));
    ASSERT_EQ("model.bin", path);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("GNA"));

}  // namespace
