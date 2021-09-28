// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "../op_reference/base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace reference_tests;
namespace {

struct RefPreprocessParams {
    RefPreprocessParams(const std::string& val): name(val) {}
        std::function<std::shared_ptr<ov::Function>()> function;
        std::vector<Tensor> inputs;
        std::vector<Tensor> expected;
        std::string name;
};

class ReferencePreprocessTest : public testing::TestWithParam<RefPreprocessParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = params.function();
        for (const auto& inp : params.inputs) {
            inputData.push_back(inp.data);
        }
        for (const auto& exp : params.expected) {
            refOutData.push_back(exp.data);
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RefPreprocessParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "name=" << param.name;
        return result.str();
    }
};


TEST_P(ReferencePreprocessTest, CompareWithHardcodedRefs) {
    Exec();
}
} // namespace

static std::shared_ptr<Function> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto res = std::make_shared<op::v0::Result>(data1);
    res->set_friendly_name("Result");
    return std::make_shared<Function>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Function> create_2inputs(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto data2 = std::make_shared<op::v0::Parameter>(type, shape);
    data2->set_friendly_name("input2");
    data1->get_output_tensor(0).set_names({"tensor_input2"});
    auto res1 = std::make_shared<op::v0::Result>(data1);
    res1->set_friendly_name("Result1");
    auto res2 = std::make_shared<op::v0::Result>(data2);
    res2->set_friendly_name("Result2");
    return std::make_shared<Function>(ResultVector{res1, res2}, ParameterVector{data1, data2});
}

static RefPreprocessParams simple_mean_scale() {
    RefPreprocessParams res("simple_mean_scale");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
        f = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean(1.f).scale(2.f))).build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 2, 2}, element::f32, std::vector<float>{1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23.});
    res.expected.emplace_back(Shape{1, 3, 2, 2}, element::f32, std::vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});
    return res;
}

static RefPreprocessParams scale_then_mean() {
    RefPreprocessParams res("scale_then_mean");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
        f = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().scale(2.0f).mean(2.0f))).build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 2, 2}, element::f32, std::vector<float>{2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 100., 200.});
    res.expected.emplace_back(Shape{1, 3, 2, 2}, element::f32, std::vector<float>{-1., 0, 1., 2., 3., 4., 5., 6., 7., 8., 48., 98.});
    return res;
}

static RefPreprocessParams convert_only() {
    RefPreprocessParams res("convert_only");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 1, 2, 2});
        f = PrePostProcessor().input(InputInfo()
                .tensor(InputTensorInfo().set_element_type(element::i16))
                .preprocess(PreProcessSteps()
                .convert_element_type(element::f32)
                .scale(3.f)
                .convert_element_type(element::u8)
                .convert_element_type(element::f32)))
                        .build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 1, 2, 2}, element::i16, std::vector<int16_t>{2, 3, 4, 5});
    res.expected.emplace_back(Shape{1, 1, 2, 2}, element::f32, std::vector<float>{0., 1., 1., 1.});
    return res;
}

static RefPreprocessParams convert_element_type_and_scale() {
    RefPreprocessParams res("convert_element_type_and_scale");
    res.function = []() {
        auto f = create_simple_function(element::u8, Shape{1, 3, 2, 2});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_element_type(element::i16))
                               .preprocess(PreProcessSteps()
                                                   .convert_element_type(element::f32)
                                                   .scale(2.f)
                                                   .convert_element_type(element::u8)))
                .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 2, 2}, element::i16,
                            std::vector<int16_t>{2, 3, 6, 8, 10, 12, 14, 16, 18, 20, 10000, 200});
    res.expected.emplace_back(Shape{1, 3, 2, 2}, element::u8,
                              std::vector<uint8_t>{1, 1, 3, 4, 5, 6, 7, 8, 9, 10, (uint8_t)5000, 100});
    return res;
}

static RefPreprocessParams tensor_element_type_and_scale() {
    RefPreprocessParams res("tensor_element_type_and_scale");
    res.function = []() {
        auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
        f = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(element::f32))
                           .preprocess(PreProcessSteps().scale(2.0f).convert_element_type(element::i8)))
            .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{2., 4., 6.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i8, std::vector<int8_t>{1, 2, 3});
    return res;
}

static RefPreprocessParams custom_preprocessing() {
    RefPreprocessParams res("custom_preprocessing");
    res.function = []() {
        auto f = create_simple_function(element::i32, Shape{1, 3, 1, 1});
        f = PrePostProcessor()
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const std::shared_ptr<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node->get_friendly_name() + "/abs");
                return abs;
            })))
            .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::i32, std::vector<int32_t>{0, 4, -6});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i32, std::vector<int32_t>{0, 4, 6});
    return res;
}

static RefPreprocessParams test_lvalue() {
    RefPreprocessParams res("test_lvalue");
    res.function = []() {
        auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
        auto p = PrePostProcessor();
        auto p1 = std::move(p);
        p = std::move(p1);
        auto inputInfo = InputInfo();
        auto inputInfo2 = std::move(inputInfo);
        inputInfo = std::move(inputInfo2);
        {
            auto inputTensorInfo = InputTensorInfo();
            auto inputTensorInfo2 = std::move(inputTensorInfo);
            inputTensorInfo = std::move(inputTensorInfo2);
            auto &same = inputTensorInfo.set_element_type(element::f32);
            same.set_layout("?CHW");
            inputInfo.tensor(std::move(same));
        }
        {
            auto preprocessSteps = PreProcessSteps();
            auto preprocessSteps2 = std::move(preprocessSteps);
            preprocessSteps = std::move(preprocessSteps2);
            preprocessSteps.mean(1.f);
            preprocessSteps.scale(2.f);
            preprocessSteps.mean({1.f, 2.f, 3.f});
            preprocessSteps.scale({2.f, 3.f, 4.f});
            preprocessSteps.custom([](const std::shared_ptr<Node> &node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node->get_friendly_name() + "/abs");
                return abs;
            });
            auto &same = preprocessSteps.convert_element_type(element::i8);
            inputInfo.preprocess(std::move(same));
        }
        p.input(std::move(inputInfo));
        f = p.build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{-9., 17., -1.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i8, std::vector<int8_t>{3, 2, 1});
    return res;
}

static RefPreprocessParams test_2_inputs_basic() {
    RefPreprocessParams res("test_2_inputs_basic");
    res.function = []() {
        auto f = create_2inputs(element::f32, Shape{1, 3, 1, 1});
        { f = PrePostProcessor().input(InputInfo(1).preprocess(PreProcessSteps().mean(1.f).scale(2.0f))).build(f); }
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3., 5., 7.});
    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3., 5., 7.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3., 5., 7.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{1., 2., 3.});
    return res;
}

static RefPreprocessParams mean_scale_vector_tensor_layout() {
    RefPreprocessParams res("mean_scale_vector_tensor_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 3, 2, 1});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_layout("NC??"))
                               .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
                .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 2, 1}, element::f32, std::vector<float>{5., 1., 5., 11., 11., -1.});
    res.expected.emplace_back(Shape{1, 3, 2, 1}, element::f32, std::vector<float>{2., 0., 1., 3., 2., -1.});
    return res;
}

static RefPreprocessParams mean_scale_dynamic_layout() {
    RefPreprocessParams res("mean_scale_dynamic_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 1, 3});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_layout("N...C"))
                               .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
                .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 2, 1, 3}, element::f32, std::vector<float>{5., 2., 7., 7., 8., -1.});
    res.expected.emplace_back(Shape{1, 2, 1, 3}, element::f32, std::vector<float>{2., 0., 1., 3., 2., -1.});
    return res;
}

std::vector<RefPreprocessParams> allPreprocessTests() {
    return std::vector<RefPreprocessParams> {
        simple_mean_scale(),
        scale_then_mean(),
        convert_only(),
        convert_element_type_and_scale(),
        tensor_element_type_and_scale(),
        custom_preprocessing(),
        test_lvalue(),
        test_2_inputs_basic(),
        mean_scale_vector_tensor_layout(),
        mean_scale_dynamic_layout()
             };
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs, ReferencePreprocessTest,
                         ::testing::ValuesIn(allPreprocessTests()),
                         ReferencePreprocessTest::getTestCaseName);
