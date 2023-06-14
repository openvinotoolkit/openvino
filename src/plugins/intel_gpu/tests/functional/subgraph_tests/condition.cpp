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
    /**
     * Simple inner body with single constant value
    */
    Type01 = 1,
    /**
     * Inner body with eltwise sum
    */
    Type02 = 2,
    /**
     * Inner body with eltwise multiply
    */
    Type03 = 3,
    /**
     * Inner body with eltwise sum and pooling
     * output shape is different with type02 and type03 for same input shape
    */
    Type04 = 4
};

public:
    InnerBodyGenerator() { }

    virtual std::shared_ptr<ngraph::Function> get_function() { return _func; }
    virtual std::shared_ptr<ngraph::opset9::Parameter> get_input() { return _param; }
    virtual std::shared_ptr<ngraph::opset1::Result> get_result() { return _result; }

    // virtual void create_body(ngraph::Shape input_shape, ngraph::element::Type prc) {
    virtual void create_body(ov::PartialShape& input_shape, ngraph::element::Type prc) {
        _func = generate(input_shape, prc);
        _param = (_func->get_parameters().size() > 0)? _func->get_parameters().front() : nullptr;
        _result = _func->get_results().front();
    }

protected:
    virtual std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) = 0;

    std::shared_ptr<ngraph::Function> _func;
    std::shared_ptr<ngraph::opset9::Parameter> _param;
    std::shared_ptr<ngraph::opset1::Result> _result;
};


class InnerBodyType01 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto constantA   = ngraph::opset9::Constant::create(prc, ov::Shape{1, 1, 1, 2}, {2.0f});
        constantA->set_friendly_name("body1_constantA");
        auto constantB   = ngraph::opset9::Constant::create(prc, ov::Shape{1, 1, 1, 2}, {12.0f});
        constantB->set_friendly_name("body1_constantB");
        auto add        = std::make_shared<ngraph::opset9::Add>(constantA, constantB);
        add->set_friendly_name("body1_add");
        auto result     = std::make_shared<ngraph::opset1::Result>(add);
        result->set_friendly_name("body1_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{},
            "constant");
        return body;
    }
};

class InnerBodyType02 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto constant   = std::make_shared<ngraph::opset9::Constant>(prc, ngraph::Shape{}, 10.0f);
        constant->set_friendly_name("body2_const");
        auto data       = std::make_shared<ngraph::opset9::Parameter>(prc, input_shape);
        data->set_friendly_name("body2_data");
        auto sum        = std::make_shared<ngraph::opset9::Multiply>(data, constant);
        sum->set_friendly_name("body2_mul");
        auto result     = std::make_shared<ngraph::opset1::Result>(sum);
        result->set_friendly_name("body2_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_mul");
        return body;
    }
};

class InnerBodyType03 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto constant   = std::make_shared<ngraph::opset9::Constant>(prc, ngraph::Shape{}, 2.0f);
        constant->set_friendly_name("body3_constant");
        auto data       = std::make_shared<ngraph::opset9::Parameter>(prc, input_shape);
        data->set_friendly_name("body3_data");
        auto add        = std::make_shared<ngraph::opset9::Add>(data, constant);
        add->set_friendly_name("body3_add");
        auto result     = std::make_shared<ngraph::opset1::Result>(add);
        result->set_friendly_name("body3_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_sum");
        return body;
    }
};

class InnerBodyType04 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto scale      = std::make_shared<ngraph::opset9::Constant>(prc, ngraph::Shape{}, 2.0f);
        scale->set_friendly_name("body4_scale");
        auto data       = std::make_shared<ngraph::opset9::Parameter>(prc, input_shape);
        data->set_friendly_name("body4_data");
        auto mul        = std::make_shared<ngraph::opset9::Multiply>(data, scale);
        mul->set_friendly_name("body4_mul");
        auto pooling    = ngraph::builder::makePooling(mul, {1, 2}, {}, {}, {1, 2},
                                                    ov::op::RoundingType::FLOOR,
                                                    ov::op::PadType::AUTO,
                                                    false,
                                                    ngraph::helpers::PoolingTypes::AVG);
        pooling->set_friendly_name("body4_pool");
        auto result     = std::make_shared<ngraph::opset1::Result>(pooling);
        result->set_friendly_name("body4_result");
        auto body       = std::make_shared<ngraph::Function>(
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
                        ngraph::element::Type prc,
                        ov::PartialShape input_shape,
                        bool cond_execution_value = false) {
                            body_then_generator = get_inner_body_generator(then_body_type);
                            body_else_generator = get_inner_body_generator(else_body_type);
                            body_then_generator->create_body(input_shape, prc);
                            body_else_generator->create_body(input_shape, prc);

                            body_else_generator->get_function()->set_friendly_name("else_inner_body");
                            body_then_generator->get_function()->set_friendly_name("then_inner_body");

                            ngraph::ParameterVector params{};
                            auto exec_cond = create_cond_execution(cond_type, params, ngraph::element::boolean, ngraph::Shape{}, cond_execution_value);
                            exec_cond->set_friendly_name("if_condition");
                            auto data = create_condition_input(params, prc, input_shape);
                            data->set_friendly_name("input_data");
                            auto cond = std::make_shared<ngraph::opset9::If>(exec_cond);
                            cond->set_friendly_name("if_operator");
                            cond->set_else_body(body_else_generator->get_function());
                            cond->set_then_body(body_then_generator->get_function());
                            cond->set_input(data, body_then_generator->get_input(), body_else_generator->get_input());
                            cond->set_output(body_then_generator->get_result(), body_else_generator->get_result());
                            auto result = std::make_shared<ngraph::opset1::Result>(cond);
                            result->set_friendly_name("outer_result");
                            function = std::make_shared<ngraph::Function>(ngraph::OutputVector {result}, params);
                        }
    std::shared_ptr<ngraph::Function> get_function() { return function; }

private:
    std::shared_ptr<ngraph::Node> create_condition_input(ngraph::ParameterVector& params,
        const ngraph::element::Type prc, const ov::PartialShape& shape,
        int value = 0, bool is_static = false) {
        if (is_static)
            return std::make_shared<ngraph::opset9::Constant>(prc, shape.to_shape(), value);

        auto input = std::make_shared<ngraph::opset9::Parameter>(prc, shape);
        params.push_back(input);
        return input;
    }

    std::shared_ptr<ngraph::Node> create_cond_execution(CondTypes cond_type,
                                                        ngraph::ParameterVector& params,
                                                        const ngraph::element::Type prc = ngraph::element::u8,
                                                        const ngraph::Shape shape = ngraph::Shape{},
                                                        bool value = false) {
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
                auto param_cond = create_condition_input(params, prc, shape);
                param_cond->set_friendly_name("param_cond");
                auto const_cond = create_condition_input(params, prc, ngraph::Shape{}, 10, true);
                const_cond->set_friendly_name("const_cond");
                if_cond = std::make_shared<ngraph::opset3::GreaterEqual>(param_cond, const_cond);
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

static std::ostream& operator<<(std::ostream& os, const InnerBodyGenerator::InnerBodyType type) {
    switch (type) {
        case InnerBodyGenerator::InnerBodyType::Type01:
        {
            os << "Type01";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type02:
        {
            os << "Type02";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type03:
        {
            os << "Type03";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type04:
        {
            os << "Type04";
            break;
        }
        default:
        {
            os << "NONE";
            break;
        }
    }
    return os;
}

static std::ostream& operator<<(std::ostream& os, const TestModelGenerator::CondTypes type) {
    switch (type) {
        case TestModelGenerator::CondTypes::CONSTANT:
        {
            os << "CONSTANT";
            break;
        }
        case TestModelGenerator::CondTypes::PARAM:
        {
            os << "PARAM";
            break;
        }
        case TestModelGenerator::CondTypes::NODE:
        {
            os << "NODE";
            break;
        }
        default:
        {
            os << "NONE";
            break;
        }
    }
    return os;
}

using ConditionParams = typename std::tuple<
        InferenceEngine::SizeVector,    // Shape
        InferenceEngine::Precision,     // Precision
        TestModelGenerator::CondTypes,  // if condition type
        LayerTestsUtils::TargetDevice   // Device name
>;

class StaticConditionLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        InferenceEngine::SizeVector data_shape;
        InferenceEngine::Precision data_prc;
        TestModelGenerator::CondTypes if_cond;
        std::string targetDevice;

        std::tie(data_shape, data_prc, if_cond, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(data_shape) << "_";
        result << "netPRC=" << std::to_string(data_prc) << "_";
        result << "ifCond=" << if_cond << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;
        TestModelGenerator::CondTypes if_cond;
        std::tie(data_shape, data_prc, if_cond, targetDevice) = GetParam();
        const auto ngShape = ov::PartialShape{data_shape};
        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);
        TestModelGenerator model_generator(InnerBodyGenerator::InnerBodyType::Type02,
                                            InnerBodyGenerator::InnerBodyType::Type03,
                                            TestModelGenerator::CondTypes::PARAM,
                                            prc,
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

    InferenceEngine::SizeVector data_shape;
    InferenceEngine::Precision data_prc;
};

TEST_P(StaticConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}

// TODO: support fp16 case
// TODO: support int8 case
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

std::vector<InferenceEngine::SizeVector> inputs_shape = {
    {3, 6}
};

std::vector<LayerTestsDefinitions::TestModelGenerator::CondTypes> if_cond_types = {
    LayerTestsDefinitions::TestModelGenerator::CondTypes::CONSTANT,
    LayerTestsDefinitions::TestModelGenerator::CondTypes::PARAM
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest, StaticConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(inputs_shape),
                    testing::ValuesIn(netPrecisions),
                    testing::ValuesIn(if_cond_types),
                    testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),
                StaticConditionLayerGPUTest::getTestCaseName);


/// Dynamic shape test

}   // namespace LayerTestsDefinitions


namespace GPULayerTestsDefinitions {

using namespace LayerTestsDefinitions;

struct CondShapeParams {
    InputShape inputShapes;
    InputShape condShapes;
};

struct InnerBodyTypeParams {
    InnerBodyGenerator::InnerBodyType then_body_type;
    InnerBodyGenerator::InnerBodyType else_body_type;
};

using ConditionGPUParams = typename std::tuple<
        CondShapeParams,                // Input Shapes
        InnerBodyTypeParams,            // Inner body type
        InferenceEngine::Precision,     // Precision
        TestModelGenerator::CondTypes,  // if condition type
        bool,                           // cond execution value
        LayerTestsUtils::TargetDevice   // Device name
>;

class ConditionLayerGPUTest : public testing::WithParamInterface<ConditionGPUParams>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionGPUParams>& obj) {
        CondShapeParams shapes;
        InnerBodyTypeParams bodyParams;
        InferenceEngine::Precision dataPrc;
        TestModelGenerator::CondTypes condType;
        bool condExecutionValue;
        std::string targetDevice;

        std::tie(shapes, bodyParams, dataPrc, condType, condExecutionValue, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << CommonTestUtils::partialShape2str({shapes.inputShapes.first}) << "_";
        for (size_t i = 0lu; i < shapes.inputShapes.second.size(); i++) {
            result << "{";
            result << CommonTestUtils::vec2str(shapes.inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_TS=(";
        result << CommonTestUtils::partialShape2str({shapes.condShapes.first}) << "_";
        for (size_t i = 0lu; i < shapes.condShapes.second.size(); i++) {
            result << "{";
            result << CommonTestUtils::vec2str(shapes.condShapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "innerBody={" << bodyParams.then_body_type << ", " << bodyParams.else_body_type << "}_";
        result << "netPRC=" << std::to_string(dataPrc) << "_";
        result << "ifCond=" << condType << "_";
        result << "targetDevice=" << targetDevice << "_";
        result << "condExecutionValue=" << (condExecutionValue?"True":"False") << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        CondShapeParams shapes;
        InnerBodyTypeParams bodyParams;
        InferenceEngine::Precision dataPrc;
        TestModelGenerator::CondTypes condType;
        bool condExecutionValue;
        std::tie(shapes, bodyParams, dataPrc, condType, condExecutionValue, targetDevice) = GetParam();

        if (condType == TestModelGenerator::CondTypes::CONSTANT) {
            init_input_shapes({shapes.inputShapes});
        } else {
            init_input_shapes({shapes.condShapes, shapes.inputShapes});
        }

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrc);
        TestModelGenerator model_generator(bodyParams.then_body_type,
                                            bodyParams.else_body_type,
                                            condType,
                                            prc,
                                            shapes.inputShapes.first,
                                            condExecutionValue);
        function = model_generator.get_function();
        function->set_friendly_name("if_operator_outer");
    }
};

TEST_P(ConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::vector<CondShapeParams> dynamicInputShapes = {
    {
        ov::test::InputShape(ov::PartialShape({1, 1, -1, -1}), {{1, 1, 8, 4}, {1, 1, 24, 16}, {1, 1, 64, 32}, {1, 1, 8, 4}, {1, 1, 24, 16}}),
        ov::test::InputShape(ov::PartialShape({1}), {{1}, {1}, {1}, {1}, {1}})
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes = {
    {
        InnerBodyGenerator::InnerBodyType::Type01,
        InnerBodyGenerator::InnerBodyType::Type02
    },
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type03
    },
    {
        InnerBodyGenerator::InnerBodyType::Type04,
        InnerBodyGenerator::InnerBodyType::Type03
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_for_constant = {
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type03
    },
    {
        InnerBodyGenerator::InnerBodyType::Type04,
        InnerBodyGenerator::InnerBodyType::Type03
    }
};

const std::vector<TestModelGenerator::CondTypes> condTypes = {
    TestModelGenerator::CondTypes::PARAM,
    TestModelGenerator::CondTypes::NODE
};

const std::vector<bool> condExecutionValues = {
    true, false
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic01, ConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes),                          // input shapes
                    testing::ValuesIn(innerBodyTypes),                              // inner body type
                    testing::ValuesIn(netPrecisions),                               // network precision
                    testing::ValuesIn(condTypes),                                   // cond type
                    testing::ValuesIn(condExecutionValues),                         // cond execution value
                    testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),     // device type
                ConditionLayerGPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic02, ConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes),                          // input shapes
                    testing::ValuesIn(innerBodyTypes_for_constant),                 // inner body type
                    testing::ValuesIn(netPrecisions),                               // network precision
                    testing::ValuesIn({TestModelGenerator::CondTypes::CONSTANT}),   // cond type
                    testing::ValuesIn(condExecutionValues),                         // cond execution value
                    testing::Values<std::string>(CommonTestUtils::DEVICE_GPU)),     // device type
                ConditionLayerGPUTest::getTestCaseName);
} // namespace GPULayerTestsDefinitions
