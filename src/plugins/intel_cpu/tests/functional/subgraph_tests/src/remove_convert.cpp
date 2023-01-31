// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace ov::test;
using namespace ngraph;
using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
class RemoveConvertCPUTest : virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        configuration.insert({"SNIPPETS_MODE", "DISABLE"});
        configuration.insert({"ENFORCE_BF16", "YES"});
        const auto input_static_shape = ov::Shape{1, 4, 32, 64};
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {},
            makeSelectedTypeStr("ref", ov::element::bf16)};
        auto in_shapes = static_shapes_to_test_representation({input_static_shape});
        init_input_shapes({in_shapes});
        auto input_params = builder::makeParams(element::bf16, {input_static_shape});
        auto convert = builder::makeConversion(input_params[0], element::f32, ::helpers::ConversionTypes::CONVERT);
        auto begin = builder::makeConstant(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0});
        auto end = builder::makeConstant(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 16, 0});
        auto stride = builder::makeConstant(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
        auto slice = builder::makeStridedSlice(convert,
                                               begin,
                                               end,
                                               stride,
                                               element::f32,
                                               {0, 0, 0, 0},
                                               {1, 1, 0, 1},
                                               {},
                                               {},
                                               {});
        auto convert2 = builder::makeConversion(slice, element::bf16, ::helpers::ConversionTypes::CONVERT);
        function = std::make_shared<ov::Model>(convert2, input_params, "remove_convert");
    };
};
namespace {
TEST_F(RemoveConvertCPUTest, smoke_RemoveConverts_CPU) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckPluginRelatedResults(compiledModel, "StridedSlice");
}
}  // namespace
}  // namespace SubgraphTestsDefinitions
