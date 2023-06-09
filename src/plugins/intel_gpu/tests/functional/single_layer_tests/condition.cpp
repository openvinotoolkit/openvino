// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"


using namespace InferenceEngine;
using namespace ov::test;

namespace LayerTestsDefinitions {

using ConditionParams = typename std::tuple<
        InferenceEngine::SizeVector,
        InferenceEngine::Precision,
        std::string>;


class ConditionLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        return "SimplConditionGPUTest";
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;
        const auto ngShape = ngraph::Shape{3, 2};
        const auto scalarShape = ngraph::Shape{};
        ngraph::ParameterVector params{};
        auto create_condition_input = [&params] (ngraph::element::Type prc, const ngraph::Shape &shape, int value = 0, bool is_static = false)
                -> std::shared_ptr<ngraph::Node> {
            if (is_static)
                return std::make_shared<ngraph::opset9::Constant>(prc, shape, value);

            auto input = std::make_shared<ngraph::opset9::Parameter>(prc, shape);
            params.push_back(input);
            return input;
        };

        // body else
        auto body_else_scale    = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, scalarShape, 2.0f);
        auto body_else_data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, ngShape);
        auto body_else_mul      = std::make_shared<ngraph::opset9::Multiply>(body_else_data, body_else_scale);
        auto body_else_result = std::make_shared<ngraph::opset1::Result>(body_else_mul);
        auto body_else          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {body_else_result},
            ngraph::ParameterVector{body_else_data});


        // body then
        auto body_then_adding   = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, scalarShape, 10.0f);
        auto body_then_data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, ngShape);
        auto body_then_sum      = std::make_shared<ngraph::opset9::Multiply>(body_then_data, body_then_adding);
        auto body_then_result = std::make_shared<ngraph::opset1::Result>(body_then_sum);
        auto body_then          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {body_then_result},
            ngraph::ParameterVector{body_then_data});


        auto exec_cond = create_condition_input(ngraph::element::boolean, scalarShape);
        auto data = create_condition_input(ngraph::element::f32, ngShape);
        auto cond = std::make_shared<ngraph::opset9::If>(exec_cond);
        cond->set_else_body(body_else);
        cond->set_then_body(body_then);
        cond->set_input(data, body_then_data, body_else_data);
        cond->set_output(body_then_result, body_else_result);
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector {cond, data}, params);
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        auto tensor_desc = info.getTensorDesc();
        auto blob = make_blob_with_precision(tensor_desc);
        blob->allocate();

        if (tensor_desc.getLayout() == InferenceEngine::SCALAR) {
            auto prc = tensor_desc.getPrecision();
            auto scalar_1d = CommonTestUtils::make_reshape_view(blob, {1});
            if (prc == InferenceEngine::Precision::BOOL) {
                auto mem_blob = dynamic_cast<InferenceEngine::MemoryBlob*>(blob.get());
                auto mem = mem_blob->rwmap();
                auto data_ptr = mem.as<bool*>();
                *data_ptr = false;
            } else {
                CommonTestUtils::fill_data_with_broadcast(scalar_1d, 0, {20.f});
            }
        } else {
            CommonTestUtils::fill_data_with_broadcast(blob, 0, {20.f});
        }
        return blob;
    }
};

TEST_P(ConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

std::vector<InferenceEngine::SizeVector> inputs_1 = {
    {2, 1, 4, 6}
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest, ConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(inputs_1),
                    testing::ValuesIn(netPrecisions),
                    testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                ConditionLayerGPUTest::getTestCaseName);

}   // namespace LayerTestsDefinitions