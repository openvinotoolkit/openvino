// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
        size_t,                            // Concat axis
        std::vector<size_t>,               // Input shapes
        InferenceEngine::Precision,        // Network precision
        std::string                        // Device name
> concatParamsTuple;

namespace LayerTestsDefinitions {

class ConcatMemoryTest : public testing::WithParamInterface<concatParamsTuple>,
                              public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj) {
        int axis;
        std::vector<size_t> input_shapes;
        InferenceEngine::Precision net_precision;
        std::string target_name;
        std::tie(axis, input_shapes, net_precision, target_name) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(input_shapes) << "_";
        result << "axis=" << axis << "_";
        result << "netPRC=" << net_precision.name() << "_";
        result << "trgDev=" << target_name;
        return result.str();
    }
protected:
    void SetUp() override {
        int axis;
        std::vector<size_t> input_shape;
        InferenceEngine::Precision net_precision;
        std::tie(axis, input_shape, net_precision, targetDevice) = this->GetParam();
        auto ng_prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);
        auto input = ngraph::builder::makeParams(ng_prc, { input_shape });

        auto variable = std::make_shared<ngraph::Variable>(ngraph::VariableInfo{ngraph::PartialShape::dynamic(), ngraph::element::dynamic, "v0"});
        auto mem_i = std::make_shared<ngraph::opset8::Constant>(ng_prc, input_shape);
        auto mem_r = std::make_shared<ngraph::opset8::ReadValue>(mem_i, variable);

        ngraph::OutputVector concat_input;
        concat_input.push_back(mem_r);
        concat_input.push_back(input.at(0));
        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_input, axis);

        auto mem_w = std::make_shared<ngraph::opset7::Assign>(input.at(0), variable);

        auto res = std::make_shared<ngraph::opset8::Result>(concat);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::SinkVector{mem_w}, input, "TestConcatMemory");
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        LoadNetwork();
        GenerateInputs();
        Infer();
    }
};

TEST_P(ConcatMemoryTest, CompareWithRefImpl) {
    Run();
};


const std::vector<size_t > axes = {1};
const std::vector<std::vector<size_t>> inShapes = {
    {1, 10},
    {1, 10, 10},
    {1, 10, 10, 10}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryTest, ConcatMemoryTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConcatMemoryTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
