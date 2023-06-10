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

class InnerBodyGenerator {
public:
    InnerBodyGenerator() { }

    virtual std::shared_ptr<ngraph::Function> get_function() { return _func; }
    virtual std::shared_ptr<ngraph::opset9::Parameter> get_input() { return _param; }
    virtual std::shared_ptr<ngraph::opset1::Result> get_result() { return _result; }

    virtual void create_body(ngraph::Shape input_shape) {
        _func = generate(input_shape);
        _param = _func->get_parameters().front();
        _result = _func->get_results().front();
    }

protected:
    virtual std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) = 0;


    std::shared_ptr<ngraph::Function> _func;
    std::shared_ptr<ngraph::opset9::Parameter> _param;
    std::shared_ptr<ngraph::opset1::Result> _result;
};

class EltwsieSumInnerBodyGenerator : public InnerBodyGenerator {
public:
    EltwsieSumInnerBodyGenerator() = default;
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto constant   = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, ngraph::Shape{}, 10.0f);
        constant->set_friendly_name("body1_const");
        auto data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        data->set_friendly_name("body1_data");
        auto sum      = std::make_shared<ngraph::opset9::Multiply>(data, constant);
        sum->set_friendly_name("body1_sum");
        auto result = std::make_shared<ngraph::opset1::Result>(sum);
        result->set_friendly_name("body1_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data});
        return body;
    }
};

class EltwiseMulInnerBodyGenerator : public InnerBodyGenerator {
public:
    EltwiseMulInnerBodyGenerator() = default;
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto scale    = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, ngraph::Shape{}, 2.0f);
        scale->set_friendly_name("body2_scale");
        auto data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        data->set_friendly_name("body2_data");
        auto mul      = std::make_shared<ngraph::opset9::Multiply>(data, scale);
        data->set_friendly_name("body2_mul");
        auto result = std::make_shared<ngraph::opset1::Result>(mul);
        data->set_friendly_name("body2_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data});
        return body;
    }
};
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
        EltwsieSumInnerBodyGenerator body_else_generator;
        body_else_generator.create_body(ngShape);

        // body then
        EltwiseMulInnerBodyGenerator body_then_generator;
        body_then_generator.create_body(ngShape);

        auto exec_cond = create_condition_input(ngraph::element::boolean, scalarShape);
        auto data = create_condition_input(ngraph::element::f32, ngShape);
        auto cond = std::make_shared<ngraph::opset9::If>(exec_cond);
        cond->set_else_body(body_else_generator.get_function());
        cond->set_then_body(body_then_generator.get_function());
        cond->set_input(data, body_then_generator.get_input(), body_else_generator.get_input());
        cond->set_output(body_then_generator.get_result(), body_else_generator.get_result());
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

////////////////////////////////////////////////////////////////////////////////
/// Dynamic shape
////////////////////////////////////////////////////////////////////////////////




}   // namespace LayerTestsDefinitions