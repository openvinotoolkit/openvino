// // Copyright (C) 2023 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include <tuple>
// #include <string>
// #include <vector>
// #include <memory>
// #include <shared_test_classes/base/ov_subgraph.hpp>
// #include <ngraph_functions/builders.hpp>
// #include <common_test_utils/common_utils.hpp>
// #include <common_test_utils/ov_tensor_utils.hpp>

// // [WA] TODO: remove
// #include "functional_test_utils/skip_tests_config.hpp"
// std::vector<std::string> disabledTestPatterns() { return {}; }


// #include "mixed_affinity_functions.hpp"


// using namespace ov::test;
// using namespace ngraph::helpers;

// using MixedAffinityBuilder = std::function<std::shared_ptr<MixedAffinityFunctionBase>(const std::vector<ov::PartialShape>& shapes)>;
// using MixedAffinityBuildManager = std::pair<MixedAffinityBuilder, std::string>;
// using MixedAffinityParams = typename std::tuple<
//         std::vector<ov::PartialShape>, // Input shapes
//         MixedAffinityBuildManager>;    // builder

// class MixedAffinityTest : public testing::WithParamInterface<MixedAffinityParams>, virtual public SubgraphBaseTest {
// public:
//     static std::string getTestCaseName(testing::TestParamInfo<MixedAffinityParams> obj) {
//         std::vector<ov::PartialShape> shapes;
//         MixedAffinityBuildManager builder;
//         std::tie(shapes, builder) = obj.param;

//         std::ostringstream result;
//         result << "IS=";
//         for (const auto& elem : shapes) {
//             result << elem << ",";
//         }
//         result << "builder=" << builder.second;
//         return result.str();
//     }

// protected:
//     void SetUp() override {
//         std::vector<ov::PartialShape> shapes;
//         MixedAffinityBuildManager builder;
//         std::tie(shapes, builder) = this->GetParam();

//         targetDevice = CommonTestUtils::DEVICE_CPU;
//         init_input_shapes(static_partial_shapes_to_test_representation(shapes));
//         function = builder.first(shapes)->getOriginal();
//         ov::pass::Serialize("/home/vgolubev/models/function.xml", "/home/vgolubev/models/function.bin").run_on_model(function);
//     }
// };

// TEST_P(MixedAffinityTest, CompareWithRefs) {
//     run();
// }

// namespace {
// std::vector<std::vector<ov::PartialShape>> inputs_1 = {
//     {{8, 3, 67, 67}},
// };

// std::vector<MixedAffinityBuildManager> builders_1 = {
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithBiasFunction>(shapes); }, "ConvWithBiasFunction"},
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeFunction>(shapes); }, "ConvWithTransposeFunction"},
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithReshapeFunction>(shapes); }, "ConvWithReshapeFunction"},
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithSplitAndResultFunction>(shapes); }, "ConvWithSplitAndResultFunction"},
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvWithS2BFunction>(shapes); }, "TwoConvWithS2BFunction"},
// };

// INSTANTIATE_TEST_SUITE_P(smoke_MixedAffinity_1input, MixedAffinityTest,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(inputs_1),
//                                  ::testing::ValuesIn(builders_1)),
//                          MixedAffinityTest::getTestCaseName);

// std::vector<std::vector<ov::PartialShape>> inputs_2 = {
//     {{8, 3, 56, 56}, {8, 3, 56, 56}},
//     {{8, 3, 56, 56}, {1, 3, 56, 56}},
//     {{1, 3, 56, 56}, {1, 3, 56, 56}},
// };

// std::vector<MixedAffinityBuildManager> builders_2 = {
//     {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvAndAddFunction>(shapes); }, "TwoConvAndAddFunction"},
// };

// INSTANTIATE_TEST_SUITE_P(smoke_MixedAffinity_2inputs, MixedAffinityTest,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(inputs_2),
//                                  ::testing::ValuesIn(builders_2)),
//                          MixedAffinityTest::getTestCaseName);

// }  // namespace
