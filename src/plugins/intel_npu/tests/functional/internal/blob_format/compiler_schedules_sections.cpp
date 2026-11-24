// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "hsm_format/compiler_schedules_sections.hpp"

// #include <gtest/gtest.h>

// #include <memory>
// #include <sstream>
// #include <string>
// #include <unordered_map>

// #include "common/functions.hpp"
// #include "common/npu_test_env_cfg.hpp"
// #include "intel_npu/utils/utils.hpp"
// #include "openvino/core/model_util.hpp"
// #include "openvino/core/rt_info/weightless_caching_attributes.hpp"
// #include "openvino/core/shape.hpp"
// #include "openvino/core/type/element_type.hpp"
// #include "openvino/op/add.hpp"
// #include "openvino/op/parameter.hpp"
// #include "openvino/util/codec_xor.hpp"
// #include "shared_test_classes/base/ov_behavior_test_utils.hpp"

// namespace {

// // duplicate from tests/functional/behavior/weights_separation.hpp
// // should we move it to common/utils?
// std::shared_ptr<ov::Model> createTestModelWithWCA() {
//     using namespace ov;

//     constexpr auto precision = element::f32;
//     const float weightsValue = 1.0f;
//     auto weights = std::make_shared<op::v0::Constant>(precision, Shape{5}, std::vector<float>{weightsValue});
//     auto input = std::make_shared<op::v0::Parameter>(precision, Shape{1});
//     auto add = std::make_shared<op::v1::Add>(input, weights);

//     weights->set_friendly_name("weights");
//     input->set_friendly_name("input");
//     add->set_friendly_name("add");

//     weights->get_rt_info()[WeightlessCacheAttribute::get_type_info_static()] =
//         WeightlessCacheAttribute(weights->get_byte_size(), 0, weights->get_element_type());

//     auto model = std::make_shared<ov::Model>(OutputVector{add}, ParameterVector{input}, "Simple with WCA");
//     ov::util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
//     return model;
// }

// }  // namespace

// namespace ov::test::behavior {
// using namespace ::intel_npu;

// class CompilerSchedulesSectionsTest : public OVPluginTestBase,
//                                       public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap, bool>> {
// public:
//     // maybe we can use SetUpTestSuite() for the compilation part
//     void SetUp() override {
//         SKIP_IF_CURRENT_TEST_IS_DISABLED();
//         std::pair<uint32_t, uint32_t> graphExtMutableCommandListExtPair;
//         std::string target_device;
//         ov::AnyMap configuration;
//         std::tie(target_device, configuration, is_dynamic) = this->GetParam();
//         zeGraphNpuExtVersion = graphExtMutableCommandListExtPair.first;
//         zeMutableCommandListExtVersion = graphExtMutableCommandListExtPair.second;

//         auto options = std::make_shared<::intel_npu::OptionsDesc>();
//         options->add<::intel_npu::PLATFORM>();
//         options->add<::intel_npu::COMPILER_TYPE>();
//         options->add<::intel_npu::BATCH_MODE>();
//         options->add<::intel_npu::MODEL_SERIALIZER_VERSION>();
//         npu_config = std::make_unique<::intel_npu::Config>(options);
//         for (const auto& [propertyName, propertyValue] : configuration) {
//             npu_config->update(propertyName, propertyValue.as<std::string>());
//         }

//         if (is_dynamic) {
//             npu_config->update(ov::intel_npu::compilation_mode.name(), "HostCompile_Interpreter");
//         }

//         auto zeroInitMock = std::make_shared<::intel_npu::ZeroInitStructsMock>(
//             ::intel_npu::test_constants::TARGET_ZE_DRIVER_NPU_EXT_VERSION,
//             intel_npu::test_constants::TARGET_ZE_GRAPH_NPU_EXT_VERSION,
//             ::intel_npu::test_constants::TARGET_ZE_COMMAND_QUEUE_NPU_EXT_VERSION,
//             ::intel_npu::test_constants::TARGET_ZE_PROFILING_NPU_EXT_VERSION,
//             ::intel_npu::test_constants::TARGET_ZE_CONTEXT_NPU_EXT_VERSION,
//             intel_npu::test_constants::TARGET_ZE_MUTABLE_COMMAND_LIST_EXT_VERSION);
//         zeroInitStruct = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zeroInitMock);

//         std::shared_ptr<ICompilerAdapter> compiler;
//         try {
//             compiler = npu_config->get<::intel_npu::COMPILER_TYPE>() == ov::intel_npu::CompilerType::DRIVER
//                            ? std::dynamic_pointer_cast<::intel_npu::ICompilerAdapter>(
//                                  std::make_shared<::intel_npu::DriverCompilerAdapter>(zeroInitStruct))
//                            : std::dynamic_pointer_cast<::intel_npu::ICompilerAdapter>(
//                                  std::make_shared<::intel_npu::PluginCompilerAdapter>(zeroInitStruct,
//                                                                                       ::intel_npu::makeVCLCompiler()));
//         } catch (...) {
//             GTEST_SKIP() << "Couldn't load compiler library";
//         }

//         graph = compiler->compile(createTestModelWithWCA(), *npu_config, ::intel_npu::AdapterDescriptor{});

//         OVPluginTestBase::SetUp();
//     }

//     static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, ov::AnyMap, bool>>& obj)
//     {
//         std::string targetDevice;
//         ov::AnyMap configuration;
//         bool is_dynamic;
//         std::tie(targetDevice, configuration, is_dynamic) = obj.param;
//         std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
//         std::ostringstream result;
//         result << "targetDevice=" << targetDevice << "_";
//         result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
//         if (!configuration.empty()) {
//             for (auto& configItem : configuration) {
//                 result << "configItem=" << configItem.first << "_";
//                 configItem.second.print(result);
//             }
//         }
//         result << "_isDynamic=" << is_dynamic;
//         return result.str();
//     }

// protected:
//     std::shared_ptr<IGraph> graph;
//     bool is_dynamic;
// };

// using ELFMainScheduleSectionTest = CompilerSchedulesSectionsTest;

// TEST_P(ELFMainScheduleSectionTest, DecryptThrowsIfSectionHasGraph) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(graph);
//     OV_EXPECT_THROW(section->decrypt(ov::util::codec_xor), ov::Exception, _);
// }

// TEST_P(ELFMainScheduleSectionTest, GetScheduleThrowsIfSectionHasGraph) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(graph);
//     OV_EXPECT_THROW(section->get_schedule(), ov::Exception, _);
// }

// TEST_P(ELFMainScheduleSectionTest, GetIndividualRuntimeRequirements) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(graph);
//     std::optional<std::string> requirements;
//     OV_ASSERT_NO_THROW(requirements = section->get_individual_runtime_requirements());
//     ASSERT_TRUE(requirements.has_value());
//     ASSERT_TRUE(requirements->size() >= 2);
//     ASSERT_EQ(requirements->front(), '[');
//     ASSERT_EQ(requirements->back(), ']');
// }

// TEST_P(ELFMainScheduleSectionTest, GetIndividualRuntimeRequirementsInvalidState) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(ov::Tensor());
//     OV_EXPECT_THROW(section->get_individual_runtime_requirements(), ov::Exception, _);
// }

// TEST_P(ELFMainScheduleSectionTest, SetGraph) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(graph);
//     OV_ASSERT_NO_THROW(section->set_graph(graph));
//     std::optional<std::string> requirements;
//     OV_ASSERT_NO_THROW(requirements = section->get_individual_runtime_requirements());
//     ASSERT_TRUE(requirements.has_value());
// }

// TEST_P(ELFMainScheduleSectionTest, SetGraphFailures) {
//     const auto section = std::make_shared<ELFMainScheduleSection>(graph);
//     OV_EXPECT_THROW(section->set_graph(graph), ov::Exception, _);
//     OV_EXPECT_THROW(section->set_graph(nullptr), ov::Exception, _);
// }

// using InitSchedulesSectionTest = CompilerSchedulesSectionsTest;

// }  // namespace ov::test::behavior

// using namespace ov::test::behavior;

// // TODO do we need these configs?

// const std::vector<ov::AnyMap> emptyConfig = {{}};

// const std::vector<ov::AnyMap> weightsSeparationConfig = {
//     {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN}, {ov::enable_weightless.name(),
//     true}}};

// const std::vector<ov::AnyMap> nonWeightsSeparationConfig = {
//     {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN},
//      {ov::enable_weightless.name(), false}}};

// INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
//                          CompilerSchedulesSectionsTest,
//                          ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(emptyConfig),
//                                             ::testing::ValuesIn(std::vector<bool>{true, false})),
//                          CompilerSchedulesSectionsTest::getTestCaseName);
