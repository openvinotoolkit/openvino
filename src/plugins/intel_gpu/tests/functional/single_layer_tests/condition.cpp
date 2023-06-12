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
    using ptr = std::shared_ptr<InnerBodyGenerator>;

enum InnerBodyType {
    Type01 = 1,
    Type02 = 2,
    Type03 = 3,
    Type04 = 4
};

public:
    InnerBodyGenerator() { }

    virtual std::shared_ptr<ngraph::Function> get_function() { return _func; }
    virtual std::shared_ptr<ngraph::opset9::Parameter> get_input() { return _param; }
    virtual std::shared_ptr<ngraph::opset1::Result> get_result() { return _result; }

    virtual void create_body(ngraph::Shape input_shape) {
        _func = generate(input_shape);
        _param = (_func->get_parameters().size() > 0)? _func->get_parameters().front() : nullptr;
        _result = _func->get_results().front();
    }

protected:
    virtual std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) = 0;

    std::shared_ptr<ngraph::Function> _func;
    std::shared_ptr<ngraph::opset9::Parameter> _param;
    std::shared_ptr<ngraph::opset1::Result> _result;
};


class InnerBodyType01 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto constant    = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, input_shape, 2.0f);
        constant->set_friendly_name("body1_constant");
        auto result = std::make_shared<ngraph::opset1::Result>(constant);
        result->set_friendly_name("body1_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{},
            "eltwise_mul");
        return body;
    }
};

class InnerBodyType02 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto constant   = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, ngraph::Shape{}, 10.0f);
        constant->set_friendly_name("body2_const");
        auto data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        data->set_friendly_name("body2_data");
        auto sum      = std::make_shared<ngraph::opset9::Multiply>(data, constant);
        sum->set_friendly_name("body2_sum");
        auto result = std::make_shared<ngraph::opset1::Result>(sum);
        result->set_friendly_name("body2_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_sum");
        return body;
    }
};

class InnerBodyType03 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto constant    = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, ngraph::Shape{}, 2.0f);
        constant->set_friendly_name("body3_constant");
        auto data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        data->set_friendly_name("body3_data");
        auto add      = std::make_shared<ngraph::opset9::Add>(data, constant);
        add->set_friendly_name("body3_add");
        auto result = std::make_shared<ngraph::opset1::Result>(add);
        result->set_friendly_name("body3_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_sum");
        return body;
    }
};

class InnerBodyType04 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ngraph::Shape input_shape) override {
        auto scale    = std::make_shared<ngraph::opset9::Constant>(ngraph::element::f32, ngraph::Shape{}, 2.0f);
        scale->set_friendly_name("body4_scale");
        auto data     = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        data->set_friendly_name("body4_data");
        auto mul      = std::make_shared<ngraph::opset9::Multiply>(data, scale);
        mul->set_friendly_name("body4_mul");
        auto pooling = ngraph::builder::makePooling(mul, {1, 2}, {}, {}, {1, 2},
                                                    ov::op::RoundingType::FLOOR,
                                                    ov::op::PadType::AUTO,
                                                    false,
                                                    ngraph::helpers::PoolingTypes::AVG);
        pooling->set_friendly_name("body4_pool");
        auto result = std::make_shared<ngraph::opset1::Result>(pooling);
        result->set_friendly_name("body4_result");
        auto body          = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_mul_pooling");
        return body;
    }
};

static std::shared_ptr<InnerBodyGenerator> get_inner_body_generator(InnerBodyGenerator::InnerBodyType type) {
    std::shared_ptr<InnerBodyGenerator> generator_ptr;
    switch (type) {
        case InnerBodyGenerator::InnerBodyType::Type01:
        {
            return std::make_shared<InnerBodyType01>();
        }
        case InnerBodyGenerator::InnerBodyType::Type02:
        {
            return std::make_shared<InnerBodyType02>();
        }
        case InnerBodyGenerator::InnerBodyType::Type03:
        {
            return std::make_shared<InnerBodyType03>();
        }
        case InnerBodyGenerator::InnerBodyType::Type04:
        {
            return std::make_shared<InnerBodyType04>();
        }
        default:
        {
            OPENVINO_ASSERT(false, "Not supported type");
        }
    }
}

class TestModelGenerator {
public:
    enum CondTypes {
        CONSTANT,
        PARAM,
        NODE
    };

public:
    TestModelGenerator(InnerBodyGenerator::InnerBodyType then_body_type,
                        InnerBodyGenerator::InnerBodyType else_body_type,
                        CondTypes cond_type,
                        ngraph::Shape input_shape) {
                            body_then_generator = get_inner_body_generator(then_body_type);
                            body_else_generator = get_inner_body_generator(else_body_type);
                            body_then_generator->create_body(input_shape);
                            body_else_generator->create_body(input_shape);

                            ngraph::ParameterVector params{};
                            auto exec_cond = create_cond_execution(cond_type, params, ngraph::element::boolean);
                            auto data = create_condition_input(params, ngraph::element::f32, input_shape);
                            auto cond = std::make_shared<ngraph::opset9::If>(exec_cond);
                            cond->set_else_body(body_else_generator->get_function());
                            cond->set_then_body(body_then_generator->get_function());
                            cond->set_input(data, body_then_generator->get_input(), body_else_generator->get_input());
                            cond->set_output(body_then_generator->get_result(), body_else_generator->get_result());
                            function = std::make_shared<ngraph::Function>(ngraph::OutputVector {cond}, params);
                        }
    std::shared_ptr<ngraph::Function> get_function() { return function; }

private:
    std::shared_ptr<ngraph::Node> create_condition_input(ngraph::ParameterVector& params,
        const ngraph::element::Type prc, const ngraph::Shape shape,
        int value = 0, bool is_static = false) {
        if (is_static)
            return std::make_shared<ngraph::opset9::Constant>(prc, shape, value);

        auto input = std::make_shared<ngraph::opset9::Parameter>(prc, shape);
        params.push_back(input);
        return input;
    }

    std::shared_ptr<ngraph::Node> create_cond_execution(CondTypes cond_type,
                                                        ngraph::ParameterVector& params,
                                                        const ngraph::element::Type prc = ngraph::element::u8,
                                                        const ngraph::Shape shape = ngraph::Shape{},
                                                        int value = 0) {
        std::shared_ptr<ngraph::Node> if_cond;
        switch (cond_type) {
            case CondTypes::CONSTANT:
            {
                if_cond = create_condition_input(params, prc, shape, value, true);
                break;
            }
            case CondTypes::PARAM:
            {
                if_cond = create_condition_input(params, prc, shape);
                break;
            }
            case CondTypes::NODE:
            {
                auto const_cond = create_condition_input(params, prc, ngraph::Shape{}, 10, true);
                const_cond->set_friendly_name("const_cond");
                auto param_cond = create_condition_input(params, prc, shape);
                param_cond->set_friendly_name("param_cond");
                auto if_cond = std::make_shared<ngraph::opset3::GreaterEqual>(param_cond, const_cond);
                if_cond->set_friendly_name("if_cond");
                break;
            }
            default:
            {
                OPENVINO_ASSERT(false, "Not supported type");
            }
        }
        return if_cond;
    }

private:
    std::shared_ptr<ngraph::Function> function;
    InnerBodyGenerator::ptr body_then_generator;
    InnerBodyGenerator::ptr body_else_generator;
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
        TestModelGenerator model_generator(InnerBodyGenerator::InnerBodyType::Type02,
                                            InnerBodyGenerator::InnerBodyType::Type03,
                                            TestModelGenerator::CondTypes::PARAM,
                                            ngShape);
        function = model_generator.get_function();
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