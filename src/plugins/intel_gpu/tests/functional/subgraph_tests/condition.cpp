// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "openvino/pass/constant_folding.hpp"


using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

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
    Type04 = 4,
    /**
     * Inner body with nested condition case
     */
    Type05 = 5,
    /**
     * Inner body with single constant with zero dimensions
     */
    Type06 = 6
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
        auto constantA   = ngraph::opset9::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 2), {2.0f});
        constantA->set_friendly_name("body1_constantA");
        auto constantB   = ngraph::opset9::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 2), {12.0f});
        constantB->set_friendly_name("body1_constantB");
        auto add        = std::make_shared<ngraph::opset9::Add>(constantA, constantB);
        add->set_friendly_name("body1_add");
        auto result     = std::make_shared<ngraph::opset1::Result>(add);
        auto o_layout = result->get_layout();
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
        auto pooling    = generate_pooling(mul, input_shape);
        pooling->set_friendly_name("body4_pool");
        auto result     = std::make_shared<ngraph::opset1::Result>(pooling);
        result->set_friendly_name("body4_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_mul_pooling");
        return body;
    }


    struct poolSpecificParams {
            ngraph::helpers::PoolingTypes   pooling_type;   // Pooling type, max or avg
            std::vector<size_t>             kernel_size;    // Kernel size
            std::vector<size_t>             stride;         // Stride
            std::vector<size_t>             pad_begin;      // Pad begin
            std::vector<size_t>             pad_end;        // Pad end
            ngraph::op::RoundingType        rounding_type;  // Rounding type
            ngraph::op::PadType             pad_type;       // Pad type
            bool                            exclued_pad;    // Exclude pad
    };

    std::shared_ptr<ov::Node> generate_pooling(const ngraph::Output<ov::Node> &in, ov::PartialShape& input_shape) {
        poolSpecificParams params;
        switch (input_shape.rank().get_length()) {
            case 5:
            {
                params = poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX,
                                                    {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                    ngraph::op::RoundingType::CEIL,
                                                    ngraph::op::PadType::SAME_LOWER, true };
                break;
            }
            case 4:
            {
                params = poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX,
                                                    {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                                    ngraph::op::RoundingType::CEIL,
                                                    ngraph::op::PadType::SAME_LOWER, true };
                break;
            }
            case 3:
            {
                params = poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX,
                                                    {2}, {2}, {0}, {0},
                                                    ngraph::op::RoundingType::CEIL,
                                                    ngraph::op::PadType::SAME_LOWER, true };
                break;
            }
            default:
            {
                OPENVINO_ASSERT(false, "Not allowed other rank");
            }
        }
        return ngraph::builder::makePooling(in, params.stride, params.pad_begin,
                                            params.pad_end, params.kernel_size, params.rounding_type,
                                            params.pad_type, params.exclued_pad, params.pooling_type);
    }
};

class InnerBodyType05 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto constant   = std::make_shared<ngraph::opset9::Constant>(prc, ngraph::Shape{}, 2.0f);
        constant->set_friendly_name("body5_constant");
        auto data       = std::make_shared<ngraph::opset9::Parameter>(prc, input_shape);
        data->set_friendly_name("body5_data");
        auto add        = std::make_shared<ngraph::opset9::Add>(data, constant);
        add->set_friendly_name("body5_add");
        std::vector<int> axes;
        for (int i = 0, r = 0; i < input_shape.rank().get_length(); i++) {
            axes.push_back(r--);
        }
        std::vector<size_t> shapeAxes;
        shapeAxes.push_back(axes.size());

        auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

        const auto reduce = ngraph::builder::makeReduce(add, reductionAxesNode, false, ngraph::helpers::ReductionType::Min);
        reduce->set_friendly_name("body5_reduce");
        auto constant_ref   = std::make_shared<ngraph::opset9::Constant>(prc, ngraph::Shape{}, 10.0f);
        constant_ref->set_friendly_name("body5_ref_constant");

        auto pred = std::make_shared<ngraph::opset3::GreaterEqual>(reduce, constant_ref);
        pred->set_friendly_name("nested_pred");

        auto nested_body_then_generator = std::make_shared<InnerBodyType03>();
        auto nested_body_else_generator = std::make_shared<InnerBodyType04>();

        auto nested_input_shape = add->get_output_partial_shape(0);
        nested_body_then_generator->create_body(nested_input_shape, prc);
        nested_body_else_generator->create_body(nested_input_shape, prc);
        nested_body_then_generator->get_function()->set_friendly_name("nested_then_inner_body");
        nested_body_else_generator->get_function()->set_friendly_name("nested_else_inner_body");

        auto cond_nested = std::make_shared<ngraph::opset8::If>(pred);
        cond_nested->set_friendly_name("if_operator_nested");
        cond_nested->set_else_body(nested_body_else_generator->get_function());
        cond_nested->set_then_body(nested_body_then_generator->get_function());
        cond_nested->set_input(add, nested_body_then_generator->get_input(), nested_body_else_generator->get_input());
        cond_nested->set_output(nested_body_then_generator->get_result(), nested_body_else_generator->get_result());

        auto result     = std::make_shared<ngraph::opset1::Result>(cond_nested);
        result->set_friendly_name("body5_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{data},
            "eltwise_sum");
        return body;
    }
};

class InnerBodyType06 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ngraph::Function> generate(ov::PartialShape& input_shape, ngraph::element::Type prc) override {
        auto constant   = ngraph::opset9::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 0), {2.0f});
        constant->set_friendly_name("body1_constant");
        // constant->get_rt_info().emplace(ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding{});
        // constant->get_rt_info().emplace("can_be_folded", false);
        auto result     = std::make_shared<ngraph::opset1::Result>(constant);
        auto o_layout = result->get_layout();
        result->set_friendly_name("body1_result");
        auto body       = std::make_shared<ngraph::Function>(
            ngraph::OutputVector {result},
            ngraph::ParameterVector{},
            "constant_only");
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
        case InnerBodyGenerator::InnerBodyType::Type05:
        {
            return std::make_shared<InnerBodyType05>();
        }
        case InnerBodyGenerator::InnerBodyType::Type06:
        {
            return std::make_shared<InnerBodyType06>();
        }
        default:
        {
            OPENVINO_ASSERT(false, "Not supported type");
        }
    }
}

class TestModelGenerator {
public:
    enum PredicateTypes {
        PARAM,
        NODE
    };

public:
    TestModelGenerator(InnerBodyGenerator::InnerBodyType then_body_type,
                        InnerBodyGenerator::InnerBodyType else_body_type,
                        PredicateTypes pred_type,
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
                            auto predicate = create_cond_execution(pred_type, params, ngraph::element::boolean, ngraph::Shape{});
                            predicate->set_friendly_name("if_predicate");
                            auto data = create_condition_input(params, prc, input_shape);
                            data->set_friendly_name("input_data");
                            auto cond = std::make_shared<ngraph::opset8::If>(predicate);
                            cond->set_friendly_name("if_operator");
                            cond->set_else_body(body_else_generator->get_function());
                            cond->set_then_body(body_then_generator->get_function());
                            cond->set_input(data, body_then_generator->get_input(), body_else_generator->get_input());
                            cond->set_output(body_then_generator->get_result(), body_else_generator->get_result());
                            if (then_body_type == InnerBodyGenerator::InnerBodyType::Type06 || else_body_type == InnerBodyGenerator::InnerBodyType::Type06) {
                                auto constant = create_condition_input(params, prc, ngraph::Shape{1}, 0, true);
                                auto addition = std::make_shared<ngraph::opset9::Add>(cond, constant);
                                auto shapeof1 = std::make_shared<ngraph::opset9::ShapeOf>(addition);
                                auto convert = std::make_shared<ngraph::opset9::Convert>(shapeof1, prc);
                                auto mul = std::make_shared<ngraph::opset9::Multiply>(convert, constant);
                                auto shapePatternsNode = create_condition_input(params, ov::element::Type_t::i64, ngraph::Shape{1}, 0, true);
                                auto reshapeOp = std::make_shared<ngraph::opset1::Reshape>(mul, shapePatternsNode, true);
                                auto result = std::make_shared<ngraph::opset1::Result>(reshapeOp);
                                result->set_friendly_name("outer_result");
                                function = std::make_shared<ngraph::Function>(ngraph::OutputVector {result}, params);
                            } else {
                                auto result = std::make_shared<ngraph::opset1::Result>(cond);
                                result->set_friendly_name("outer_result");
                                function = std::make_shared<ngraph::Function>(ngraph::OutputVector {result}, params);
                            }
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

    std::shared_ptr<ngraph::Node> create_cond_execution(PredicateTypes pred_type,
                                                        ngraph::ParameterVector& params,
                                                        const ngraph::element::Type prc = ngraph::element::u8,
                                                        const ngraph::Shape shape = ngraph::Shape{}) {
        std::shared_ptr<ngraph::Node> pred;
        switch (pred_type) {
            case PredicateTypes::PARAM:
            {
                pred = create_condition_input(params, prc, shape);
                break;
            }
            case PredicateTypes::NODE:
            {
                auto param_cond = create_condition_input(params, prc, shape);
                param_cond->set_friendly_name("param_cond");
                auto const_cond = create_condition_input(params, prc, ngraph::Shape{}, 1, true);
                const_cond->set_friendly_name("const_cond");
                pred = std::make_shared<ngraph::opset3::GreaterEqual>(param_cond, const_cond);
                pred->set_friendly_name("pred");
                break;
            }
            default:
            {
                OPENVINO_ASSERT(false, "Not supported type");
            }
        }
        return pred;
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
        case InnerBodyGenerator::InnerBodyType::Type05:
        {
            os << "Type05";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type06:
        {
            os << "Type06";
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

static std::ostream& operator<<(std::ostream& os, const TestModelGenerator::PredicateTypes type) {
    switch (type) {
        case TestModelGenerator::PredicateTypes::PARAM:
        {
            os << "PARAM";
            break;
        }
        case TestModelGenerator::PredicateTypes::NODE:
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
        TestModelGenerator::PredicateTypes,  // if predicate type
        LayerTestsUtils::TargetDevice   // Device name
>;

class StaticConditionLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        InferenceEngine::SizeVector data_shape;
        InferenceEngine::Precision data_prc;
        TestModelGenerator::PredicateTypes pred;
        std::string targetDevice;

        std::tie(data_shape, data_prc, pred, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "netPRC=" << std::to_string(data_prc) << "_";
        result << "ifCond=" << pred << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        TestModelGenerator::PredicateTypes pred;
        std::tie(data_shape, data_prc, pred, targetDevice) = GetParam();
        const auto ngShape = ov::PartialShape{data_shape};
        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);
        TestModelGenerator model_generator(InnerBodyGenerator::InnerBodyType::Type02,
                                            InnerBodyGenerator::InnerBodyType::Type03,
                                            pred,
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
            auto scalar_1d = ov::test::utils::make_reshape_view(blob, {1});
            if (prc == InferenceEngine::Precision::BOOL) {
                auto mem_blob = dynamic_cast<InferenceEngine::MemoryBlob*>(blob.get());
                auto mem = mem_blob->rwmap();
                auto data_ptr = mem.as<bool*>();
                *data_ptr = false;
            } else {
                ov::test::utils::fill_data_with_broadcast(scalar_1d, 0, {20.f});
            }
        } else {
            ov::test::utils::fill_data_with_broadcast(blob, 0, {20.f});
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

std::vector<InferenceEngine::Precision> netPrecisions_static = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I8
};

std::vector<InferenceEngine::SizeVector> inputs_shape = {
    {3, 6}
};

std::vector<GPULayerTestsDefinitions::TestModelGenerator::PredicateTypes> if_cond_types = {
    GPULayerTestsDefinitions::TestModelGenerator::PredicateTypes::PARAM
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_static, StaticConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(inputs_shape),
                    testing::ValuesIn(netPrecisions_static),
                    testing::ValuesIn(if_cond_types),
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                StaticConditionLayerGPUTest::getTestCaseName);


/// Dynamic shape test
struct InnerBodyTypeParams {
    InnerBodyGenerator::InnerBodyType then_body_type;
    InnerBodyGenerator::InnerBodyType else_body_type;
};

using ConditionGPUParams = typename std::tuple<
        InputShape,                         // Input Shapes
        InnerBodyTypeParams,                // Inner body type
        InferenceEngine::Precision,         // Precision
        TestModelGenerator::PredicateTypes, // if predicate type
        LayerTestsUtils::TargetDevice       // Device name
>;

class DynamicConditionLayerGPUTest : public testing::WithParamInterface<ConditionGPUParams>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionGPUParams>& obj) {
        InputShape inputShapes;
        InnerBodyTypeParams bodyParams;
        InferenceEngine::Precision dataPrc;
        TestModelGenerator::PredicateTypes condType;
        std::string targetDevice;

        std::tie(inputShapes, bodyParams, dataPrc, condType, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "innerBody={" << bodyParams.then_body_type << ", " << bodyParams.else_body_type << "}_";
        result << "netPRC=" << dataPrc << "_";
        result << "ifCond=" << condType << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        InputShape inputShapes;
        InnerBodyTypeParams bodyParams;
        InferenceEngine::Precision dataPrc;
        TestModelGenerator::PredicateTypes condType;
        std::tie(inputShapes, bodyParams, dataPrc, condType, targetDevice) = GetParam();
        auto num_second = inputShapes.second.size();
        std::vector<ov::Shape> condSecondVec;
        for (size_t i = 0; i < num_second; i++) {
            condSecondVec.push_back({});
        }
        auto condShapes = ov::test::InputShape(ov::PartialShape({}), condSecondVec);
        init_input_shapes({condShapes, inputShapes});

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrc);
        TestModelGenerator model_generator(bodyParams.then_body_type,
                                            bodyParams.else_body_type,
                                            condType,
                                            prc,
                                            inputShapes.first);
        function = model_generator.get_function();
        function->set_friendly_name("if_operator_outer");
    }

    /**
     * @brief Override generate_inputs to support boolean param for if(condition) operator.
     *
     * @param targetInputStaticShapes
     */
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ov::Shape input_shape;
        for (auto& shape : targetInputStaticShapes) {
            // Change condition to cover 1 dim input shape
            if (shape.size() > 0) {
                input_shape = shape;
                break;
            }
        }

        inputs.clear();
        for (const auto &param : function->get_parameters()) {
            if (param->get_output_element_type(0) == ov::element::boolean) {
                auto tensor = ov::Tensor{ov::element::boolean, {}};
                auto p_data = tensor.data<ov::element_type_traits<ov::element::boolean>::value_type>();
                p_data[0] = (niter++ % 2);

                inputs.insert({param, tensor});
            } else {
                ov::test::utils::InputGenerateData inGenData;
                inGenData.range         = 10;
                inGenData.start_from    = 0;
                inGenData.resolution    = 128;
                inGenData.seed          = 1;
                auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), input_shape, inGenData.range,
                                                                        inGenData.start_from, inGenData.resolution, inGenData.seed);
                inputs.insert({param, tensor});
            }
        }
    }

    size_t niter = 0;
};

TEST_P(DynamicConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

const std::vector<InferenceEngine::Precision> netPrecisions_f32 = {
    InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::Precision> netPrecisions_f16 = {
    InferenceEngine::Precision::FP16
};

const std::vector<ov::test::InputShape> dynamicInputShapes_f32 = {
    ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{4, 1, 1, 64, 32}, {6, 1, 1, 8, 4}, {8, 1, 1, 24, 16}}),
    ov::test::InputShape(ov::PartialShape({1, 1, -1, -1}), {{1, 1, 64, 32}, {1, 1, 8, 4}, {1, 1, 24, 16}})
};

const std::vector<ov::test::InputShape> dynamicInputShapes_f16 = {
    ov::test::InputShape(ov::PartialShape({1, 1, -1, -1}), {{1, 1, 64, 32}, {1, 1, 8, 4}, {1, 1, 24, 16}}),
    ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 24, 16}, {2, 64, 32}, {2, 8, 4}})
};

const std::vector<ov::test::InputShape> dynamicInputShapes_zero_dims = {
    ov::test::InputShape(ov::PartialShape({-1}), {{24}, {64}, {8}})
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_f32 = {
    {
        InnerBodyGenerator::InnerBodyType::Type01,
        InnerBodyGenerator::InnerBodyType::Type02
    },
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type03
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_f16 = {
    {
        InnerBodyGenerator::InnerBodyType::Type04,
        InnerBodyGenerator::InnerBodyType::Type03
    },
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type05
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_zero_dims = {
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type06
    },
};

const std::vector<TestModelGenerator::PredicateTypes> condTypes = {
    TestModelGenerator::PredicateTypes::PARAM,
    TestModelGenerator::PredicateTypes::NODE
};

const std::vector<TestModelGenerator::PredicateTypes> condTypes_zero_dims = {
    TestModelGenerator::PredicateTypes::PARAM
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic_f32, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_f32),                          // input shapes
                    testing::ValuesIn(innerBodyTypes_f32),                              // inner body type
                    testing::ValuesIn(netPrecisions_f32),                               // network precision
                    testing::ValuesIn(condTypes),                                       // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic_f16, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_f16),                          // input shapes
                    testing::ValuesIn(innerBodyTypes_f16),                              // inner body type
                    testing::ValuesIn(netPrecisions_f16),                               // network precision
                    testing::ValuesIn(condTypes),                                       // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_zero_dims, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_zero_dims),                    // input shapes
                    testing::ValuesIn(innerBodyTypes_zero_dims),                        // inner body type
                    testing::ValuesIn(netPrecisions_f32),                               // network precision
                    testing::ValuesIn(condTypes_zero_dims),                             // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);
} // namespace GPULayerTestsDefinitions
