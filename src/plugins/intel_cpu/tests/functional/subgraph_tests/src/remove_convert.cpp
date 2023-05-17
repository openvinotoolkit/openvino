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
using RemoveConvertCPUTestParams = std::tuple<ElementType, InputShape>;

class RemoveUselessBF16ConvertCPUTest : public testing::WithParamInterface<RemoveConvertCPUTestParams>,
                             virtual public SubgraphBaseTest,
                             public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoveConvertCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();
        targetDevice = CommonTestUtils::DEVICE_CPU;
        if (inType == ElementType::bf16) {
            configuration.insert({"ENFORCE_BF16", "YES"});
        }
        std::tie(inFmts, outFmts, priority, selectedType) =
            CPUSpecificParams{{}, {}, {}, makeSelectedTypeStr("ref", inType)};
        init_input_shapes({inputShape});
        auto input_params = builder::makeDynamicParams(inType, {inputShape.first});
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
        auto convert2 = builder::makeConversion(slice, inType, ::helpers::ConversionTypes::CONVERT);
        function = std::make_shared<ov::Model>(convert2, input_params, "remove_convert");
    };
};

TEST_P(RemoveUselessBF16ConvertCPUTest, CompareWithRefs) {
    run();
    //Convert is removed by graph_optimizer
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    //Disable Convert with Snippet
    CheckNumberOfNodesWithType(compiledModel, "Subgraph", 0);
    CheckPluginRelatedResults(compiledModel, "StridedSlice");
}
class SkipRemoveConvertCPUTest : public testing::WithParamInterface<RemoveConvertCPUTestParams>,
                             virtual public SubgraphBaseTest,
                             public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoveConvertCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();
        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::tie(inFmts, outFmts, priority, selectedType) =
            CPUSpecificParams{{}, {}, {}, makeSelectedTypeStr("ref", inType)};
        init_input_shapes({inputShape});
        auto inputParams1 = builder::makeDynamicParams(inType, {inputShape.first});
        auto leftAdd = builder::makeConstant<int64_t>(ElementType::i64, std::vector<size_t>{1}, {}, true);
        auto convert = builder::makeConversion(inputParams1[0], element::i64, ::helpers::ConversionTypes::CONVERT);
        auto add = std::make_shared<opset8::Add>(leftAdd, convert);
        function = std::make_shared<ov::Model>(OutputVector{add->output(0), convert->output(0)},
           inputParams1, "skip_remove_convert");
    };
};

TEST_P(SkipRemoveConvertCPUTest, CompareWithRefs) {
    run();
    //Convert is not removed by graph_optimizer
    CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
}

namespace {
const std::vector<ElementType> inPrecisions = {
    // only bf16 could match this pattern
    ElementType::bf16,
};

const std::vector<InputShape> inputShapes = {
    // dynamic batch
    {{-1, 4, 32, 64}, {{1, 4, 32, 64}, {2, 4, 32, 64}, {3, 4, 32, 64}}},
    {{-1, -1, -1, -1}, {{1, 4, 32, 64}, {2, 4, 32, 64}, {3, 4, 32, 64}}},
    // static shape
    {{1, 4, 32, 64}, {{1, 4, 32, 64}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RemoveConvert,
                         RemoveUselessBF16ConvertCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inPrecisions), ::testing::ValuesIn(inputShapes)),
                         RemoveUselessBF16ConvertCPUTest::getTestCaseName);

const std::vector<ElementType> skipRemoveinPrec = {
    //convert i32 to i64 is part of the model, not inserted by cpu precsion conversion
    ElementType::i32,
};

const std::vector<InputShape> skipRemoveInputShapes = {
    // static shape
    {{10}, {{10}}},
    // dynamic shape
    {{-1}, {{10}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_SkipRemoveConvert,
                         SkipRemoveConvertCPUTest,
                         ::testing::Combine(::testing::ValuesIn(skipRemoveinPrec), ::testing::ValuesIn(skipRemoveInputShapes)),
                         SkipRemoveConvertCPUTest::getTestCaseName);

}  // namespace
}  // namespace SubgraphTestsDefinitions
