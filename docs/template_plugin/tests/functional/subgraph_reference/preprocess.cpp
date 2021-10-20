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
        float abs_threshold = 0.01f;
        float rel_threshold = 0.01f;
        std::string name;
};

class ReferencePreprocessTest : public testing::TestWithParam<RefPreprocessParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        const auto& params = GetParam();
        function = params.function();
        for (const auto& inp : params.inputs) {
            inputData.push_back(inp.data);
        }
        for (const auto& exp : params.expected) {
            refOutData.push_back(exp.data);
        }
        abs_threshold = params.abs_threshold;
        threshold = params.rel_threshold;
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
    auto c = op::v0::Constant::create(type, {1}, {0});
    auto op = std::make_shared<op::v1::Add>(data1, c);
    op->set_friendly_name("Add0");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    return std::make_shared<ov::Function>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Function> create_2inputs(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto c1 = op::v0::Constant::create(type, {1}, {0});
    auto op1 = std::make_shared<op::v1::Add>(data1, c1);
    op1->set_friendly_name("Add01");
    auto data2 = std::make_shared<op::v0::Parameter>(type, shape);
    data2->get_output_tensor(0).set_names({"tensor_input2"});
    data2->set_friendly_name("input2");
    auto c2 = op::v0::Constant::create(type, {1}, {0});
    auto op2 = std::make_shared<op::v1::Add>(data2, c2);
    op2->set_friendly_name("Add02");
    auto res1 = std::make_shared<op::v0::Result>(op1);
    res1->set_friendly_name("Result1");
    res1->get_output_tensor(0).set_names({"tensor_output1"});
    auto res2 = std::make_shared<op::v0::Result>(op2);
    res2->set_friendly_name("Result2");
    res2->get_output_tensor(0).set_names({"tensor_output2"});
    return std::make_shared<ov::Function>(ResultVector{res1, res2}, ParameterVector{data1, data2});
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
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const Output<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
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
            preprocessSteps.custom([](const Output<Node> &node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
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
        f = PrePostProcessor().input(InputInfo(0)
                                             .preprocess(
                                                     PreProcessSteps()
                                                             .mean(1.f)))
                .input(
                        InputInfo("tensor_input2")
                                .preprocess(PreProcessSteps()
                                                    .mean(1.f)
                                                    .scale(2.0f)))
                .build(f);
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3., 5., 7.});
    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3., 5., 7.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{2., 4., 6.});
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

static RefPreprocessParams resize_to_network_height() {
    RefPreprocessParams res("resize_to_network_height");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 1, 1});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                               .network(InputNetworkInfo().set_layout("NHWC"))
                )
                .build(f);
        return f;
    };
    res.inputs.emplace_back(element::f32, Shape{1, 4, 1, 1}, std::vector<float>{0., 2., 4., 6.});
    res.expected.emplace_back(Shape{1, 2, 1, 1}, element::f32, std::vector<float>{1., 5.});
    return res;
}

static RefPreprocessParams resize_to_network_width() {
    RefPreprocessParams res("resize_to_network_width");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 2, 2});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                               .network(InputNetworkInfo().set_layout("NCHW")))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(element::f32, Shape{1, 1, 2, 6}, std::vector<float>{0., 1., 2., 3., 4., 5.,
                                                                                0., 1., 2., 3., 4., 5.});
    res.expected.emplace_back(Shape{1, 1, 2, 2}, element::f32, std::vector<float>{1., 4., 1., 4.});
    return res;
}

static RefPreprocessParams resize_from_spatial_dims() {
    RefPreprocessParams res("resize_from_spatial_dims");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 1, 1});
        auto t = InputTensorInfo();
        t.set_spatial_static_shape(1, 4);
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(std::move(t))
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_CUBIC))
                               .network(InputNetworkInfo().set_layout("NCHW")))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(element::f32, Shape{1, 1, 1, 7}, std::vector<float>{0., 0.25, 1., 2.25, 4., 6.25, 9});
    res.expected.emplace_back(Shape{1, 1, 1, 1}, element::f32, std::vector<float>{2.25});
    return res;
}

static RefPreprocessParams resize_i8() {
    RefPreprocessParams res("resize_i8");
    res.function = []() {
        auto f = create_simple_function(element::i8, PartialShape{1, 3, 1, 1});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                    .set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                               .network(InputNetworkInfo().set_layout("NCHW")))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(element::i8, Shape{1, 3, 2, 2}, std::vector<int8_t>{0, 0, 0, 0,
                                                                                1, 1, 1, 1,
                                                                                2, 2, 2, 2});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i8, std::vector<int8_t>{0, 1, 2});
    return res;
}

static RefPreprocessParams resize_to_network_width_height() {
    RefPreprocessParams res("resize_to_network_width_height");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 1, 4, 4});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_spatial_static_shape(5, 5))
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_NEAREST))
                               .network(InputNetworkInfo().set_layout("...HW")))
                .build(f);
        return f;
    };

    auto result = std::make_shared<HostTensor>();
    // clang-format off
    std::vector<float> input = {0., 1., 2., 3., 4.,
                                1., 2., 3., 4., 5.,
                                2., 3., 4., 5., 6.,
                                3., 4., 5., 6., 7.,
                                2., 3., 4., 5., 6.};
    std::vector<float> expected = {0., 1., 3., 4.,
                                   1., 2., 4., 5.,
                                   3., 4., 6., 7.,
                                   2., 3., 5., 6.};
    // clang-format on
    res.inputs.emplace_back(element::f32, Shape{1, 1, 5, 5}, input);
    res.expected.emplace_back(Shape{1, 1, 4, 4}, element::f32, expected);
    return res;
}

static RefPreprocessParams resize_to_specified_width_height() {
    RefPreprocessParams res("resize_to_specified_width_height");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 1, Dimension::dynamic(), Dimension::dynamic()});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_NEAREST, 4, 4))
                               .network(InputNetworkInfo().set_layout("...HW")))
                .build(f);
        return f;
    };

    auto result = std::make_shared<HostTensor>();
    // clang-format off
    std::vector<float> input = {0., 1., 2., 3., 4.,
                                1., 2., 3., 4., 5.,
                                2., 3., 4., 5., 6.,
                                3., 4., 5., 6., 7.,
                                2., 3., 4., 5., 6.};
    std::vector<float> expected = {0., 1., 3., 4.,
                                   1., 2., 4., 5.,
                                   3., 4., 6., 7.,
                                   2., 3., 5., 6.};
    // clang-format on
    res.inputs.emplace_back(element::f32, Shape{1, 1, 5, 5}, input);
    res.expected.emplace_back(Shape{1, 1, 4, 4}, element::f32, expected);
    return res;
}

static RefPreprocessParams resize_lvalues() {
    RefPreprocessParams res("resize_lvalues");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 1, 2});
        f->get_parameters().front()->set_layout("NCHW");
        auto t = InputTensorInfo();
        t.set_spatial_dynamic_shape();
        auto s = PreProcessSteps();
        s.resize(ResizeAlgorithm::RESIZE_LINEAR, 1, 6); // to specified shape
        s.resize(ResizeAlgorithm::RESIZE_LINEAR);  // to network's shape
        auto n = InputNetworkInfo();
        n.set_layout("NCHW");
        auto i = InputInfo();
        i.tensor(std::move(t));
        i.preprocess(std::move(s));
        i.network(std::move(n));
        f = PrePostProcessor()
                .input(std::move(i))
                .build(f);
        return f;
    };
    // clang-format off
    res.inputs.emplace_back(element::f32, Shape{1, 1, 1, 18}, std::vector<float>{0., 0., 0.,
                                                                                 1., 1., 1.,
                                                                                 2., 2., 2.,
                                                                                 3., 3., 3.,
                                                                                 4., 4., 4.,
                                                                                 5., 5., 5.});
    // clang-format on
    res.expected.emplace_back(Shape{1, 1, 2, 1}, element::f32, std::vector<float>{1., 4.});
    return res;
}

static RefPreprocessParams convert_layout_nhwc_to_nchw_lvalue() {
    RefPreprocessParams res("convert_layout_nhwc_to_nchw_lvalue");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 3, 2, 2});
        f->get_parameters()[0]->set_layout("NCHW");
        auto p = PreProcessSteps();
        p.convert_layout("NCHW");

        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_layout("NHWC"))
                               .preprocess(std::move(p)))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3}, element::u8, std::vector<uint8_t>{1,  2,  3,       // [H=0, W=0, RGB]
                                                                                 4,  5,  6,       // [H=0, W=1]
                                                                                 7,  8,  9,       // [H=1, W=0]
                                                                                 10, 11, 12});    // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2}, element::u8, std::vector<uint8_t>{1, 4, 7, 10,    // R
                                                                                   2, 5, 8, 11,    // G
                                                                                   3, 6, 9, 12});  // B
    return res;
}

static RefPreprocessParams convert_layout_nhwc_to_net_no_tensor_shape() {
    RefPreprocessParams res("convert_layout_nhwc_to_net_no_tensor_shape");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 3, 2, 2});
        f->get_parameters()[0]->set_layout("NCHW");
        auto p = PreProcessSteps();
        p.convert_layout();
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo().set_layout("NHWC"))
                               .preprocess(std::move(p)))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3}, element::u8, std::vector<uint8_t>{1,  2,  3,       // [H=0, W=0, RGB]
                                                                                 4,  5,  6,       // [H=0, W=1]
                                                                                 7,  8,  9,       // [H=1, W=0]
                                                                                 10, 11, 12});    // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2}, element::u8, std::vector<uint8_t>{1, 4, 7, 10,    // R
                                                                                   2, 5, 8, 11,    // G
                                                                                   3, 6, 9, 12});  // B
    return res;
}

static RefPreprocessParams resize_and_convert_layout() {
    RefPreprocessParams res("resize_and_convert_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 2});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                               .set_layout("NCHW")
                                               .set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps()
                                                   .resize(ResizeAlgorithm::RESIZE_LINEAR)
                                                   .convert_layout())
                               .network(InputNetworkInfo().set_layout("NHWC")))
                .build(f);
        return f;
    };

    auto result = std::make_shared<HostTensor>();
    // clang-format off
    std::vector<float> input = {
            1., 1., 1., 1., // channel 1
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., // channel 2
            2., 2., 2., 2.,
            2., 2., 2., 2.,
            2., 2., 2., 2.,
    };
    std::vector<float> expected = {1., 2., 1., 2., 1., 2., 1., 2.};
    // clang-format on
    res.inputs.emplace_back(element::f32, Shape{1, 2, 4, 4}, input);
    res.expected.emplace_back(Shape{1, 2, 2, 2}, element::f32, expected);
    return res;
}

static RefPreprocessParams convert_color_nv12_to_bgr_two_planes() {
    RefPreprocessParams res("convert_color_nv12_to_bgr_two_planes");
    res.abs_threshold = 2.f; // Allow small color conversion deviations
    res.rel_threshold = 1.f; // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::u8, PartialShape{1, 4, 4, 3});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                               .set_color_format(ColorFormat::NV12_TWO_PLANES))
                               .preprocess(PreProcessSteps()
                                                   .convert_color(ColorFormat::BGR)))
                .build(f);
        return f;
    };

    // clang-format off
    auto input_y = std::vector<uint8_t> {81, 81, 145, 145,      // RRGG
                                         81, 81, 145, 145,      // RRGG
                                         41, 41, 81, 81,        // BBRR
                                         41, 41, 81, 81};       // BBRR
    auto input_shape_y = Shape{1, 4, 4, 1};
    auto input_uv = std::vector<uint8_t> {240, 90,      // R (2x2)
                                          34, 54,       // G (2x2)
                                          110, 240,     // B (2x2)
                                          240, 90};     // R (2x2)
    auto input_shape_uv = Shape{1, 2, 2, 2};
    auto exp_out = std::vector<uint8_t> {0, 0, 255,  0, 0, 255,  0, 255, 0,  0, 255, 0,
                                         0, 0, 255,  0, 0, 255,  0, 255, 0,  0, 255, 0,
                                         255, 0, 0,  255, 0, 0,  0, 0, 255,  0, 0, 255,
                                         255, 0, 0,  255, 0, 0,  0, 0, 255,  0, 0, 255};
    auto out_shape = Shape{1, 4, 4, 3};
    // clang-format on
    res.inputs.emplace_back(element::u8, input_shape_y, input_y);
    res.inputs.emplace_back(element::u8, input_shape_uv, input_uv);
    res.expected.emplace_back(out_shape, element::u8, exp_out);
    return res;
}

static RefPreprocessParams convert_color_nv12_single_plane() {
    RefPreprocessParams res("convert_color_nv12_single_plane");
    res.abs_threshold = 2.f; // Allow small color conversion deviations
    res.rel_threshold = 1.f; // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                               .set_color_format(ColorFormat::NV12_SINGLE_PLANE))
                               .preprocess(PreProcessSteps()
                                                   .convert_color(ColorFormat::RGB)))
                .build(f);
        return f;
    };

    // clang-format off
    auto input = std::vector<float> {  81, 81, 145, 145,      // RRGG
                                       81, 81, 145, 145,      // RRGG
                                       41, 41, 81, 81,        // BBRR
                                       41, 41, 81, 81,        // BBRR
                                       240, 90, 34, 54, 110, 240, 240, 90};     // UV (RGBR)
    auto input_shape = Shape{1, 6, 4, 1};
    auto exp_out = std::vector<float> {255, 0, 0,  255, 0, 0,  0, 255, 0,  0, 255, 0,    // RRGG
                                       255, 0, 0,  255, 0, 0,  0, 255, 0,  0, 255, 0,    // RRGG
                                       0, 0, 255,  0, 0, 255,  255, 0, 0,  255, 0, 0,    // BBRR
                                       0, 0, 255,  0, 0, 255,  255, 0, 0,  255, 0, 0,    // BBRR
                                       };
    auto out_shape = Shape{1, 4, 4, 3};
    // clang-format on
    res.inputs.emplace_back(element::f32, input_shape, input);
    res.expected.emplace_back(out_shape, element::f32, exp_out);
    return res;
}

static RefPreprocessParams convert_color_nv12_layout_resize() {
    RefPreprocessParams res("convert_color_nv12_layout_resize");
    res.abs_threshold = 2.f; // Allow small color conversion deviations
    res.rel_threshold = 1.f; // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 3, 2, 2});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                               .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                                               .set_element_type(element::u8)
                                               .set_spatial_dynamic_shape())
                               .preprocess(PreProcessSteps()
                                                   .convert_color(ColorFormat::RGB)
                                                   .convert_layout()
                                                   .convert_element_type(element::f32)
                                                   .resize(ResizeAlgorithm::RESIZE_NEAREST))
                               .network(InputNetworkInfo().set_layout("NCHW")))
                .build(f);
        return f;
    };

    auto result = std::make_shared<HostTensor>();
    // clang-format off
    auto input = std::vector<uint8_t> {81, 81, 145, 145,      // RRGG
                                       81, 81, 145, 145,      // RRGG
                                       41, 41, 81, 81,        // BBRR
                                       41, 41, 81, 81,        // BBRR
                                       240, 90, 34, 54, 110, 240, 240, 90};     // UV (RGBR)
    auto input_shape = Shape{1, 6, 4, 1};
    auto exp_out = std::vector<float> {255, 0, 0, 255,     // R channel
                                       0, 255, 0, 0,       // G channel
                                       0, 0, 255, 0};      // B channel
    auto out_shape = Shape{1, 2, 2, 3};
    // clang-format on
    res.inputs.emplace_back(element::u8, input_shape, input);
    res.expected.emplace_back(out_shape, element::f32, exp_out);
    return res;
}

static RefPreprocessParams element_type_before_convert_color_nv12() {
    RefPreprocessParams res("element_type_before_convert_color_nv12");
    res.abs_threshold = 2.f; // Allow small color conversion deviations
    res.rel_threshold = 1.f; // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
        f = PrePostProcessor()
                .input(InputInfo()
                               .tensor(InputTensorInfo()
                                               .set_element_type(element::u8)
                                               .set_color_format(ColorFormat::NV12_TWO_PLANES))
                               .preprocess(PreProcessSteps()
                                                   .convert_element_type(element::f32)
                                                   .convert_color(ColorFormat::RGB))
                               .network(InputNetworkInfo().set_layout("NHWC")))
                .build(f);
        return f;
    };

    // clang-format off
    auto input_y = std::vector<uint8_t> {81, 81, 81, 81};
    auto input_shape_y = Shape{1, 2, 2, 1};
    auto input_uv = std::vector<uint8_t> {240, 90};
    auto input_shape_uv = Shape{1, 1, 1, 2};
    auto exp_out = std::vector<float> {255, 0, 0,  255, 0, 0,  255, 0, 0,  255, 0,  0};
    auto out_shape = Shape{1, 2, 2, 3};
    // clang-format on
    res.inputs.emplace_back(element::u8, input_shape_y, input_y);
    res.inputs.emplace_back(element::u8, input_shape_uv, input_uv);
    res.expected.emplace_back(out_shape, element::f32, exp_out);
    return res;
}

static RefPreprocessParams postprocess_2_inputs_basic() {
    RefPreprocessParams res("postprocess_2_inputs_basic");
    res.function = []() {
        auto f = create_2inputs(element::f32, Shape{1, 3, 1, 2});
        f = PrePostProcessor()
                .output(OutputInfo("tensor_output1")
                                .network(OutputNetworkInfo().set_layout("NCHW"))
                                .postprocess(PostProcessSteps().convert_layout())
                                .tensor(OutputTensorInfo().set_layout("NHWC")))
                .output(OutputInfo("tensor_output2")
                                .postprocess(PostProcessSteps().convert_element_type())
                                .tensor(OutputTensorInfo().set_element_type(element::u8)))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{1.1, 2.1, 3.1, 4.1, 5.1, 6.1});
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{1.1, 2.1, 3.1, 4.1, 5.1, 6.1});
    res.expected.emplace_back(Shape{1, 1, 2, 3}, element::f32, std::vector<float>{1.1, 3.1, 5.1, 2.1, 4.1, 6.1});
    res.expected.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    return res;
}

static RefPreprocessParams pre_and_post_processing() {
    RefPreprocessParams res("pre_and_post_processing");
    res.function = []() {
        auto f = create_2inputs(element::f32, Shape{1, 3, 1, 2});
        f = PrePostProcessor()
                .input(InputInfo(0)
                                .tensor(InputTensorInfo().set_element_type(element::u8))
                                .preprocess(PreProcessSteps().convert_element_type(element::f32).mean(1.f)))
                .input(InputInfo(1)
                               .preprocess(PreProcessSteps().scale(2.f)))
                .output(OutputInfo("tensor_output1")
                                .network(OutputNetworkInfo().set_layout("NCHW"))
                                .postprocess(PostProcessSteps().convert_layout())
                                .tensor(OutputTensorInfo().set_layout("NHWC")))
                .output(OutputInfo("tensor_output2")
                                .postprocess(PostProcessSteps().convert_element_type())
                                .tensor(OutputTensorInfo().set_element_type(element::u8)))
                .build(f);
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{2.2, 4.2, 6.2, 2.4, 4.4, 6.4});
    res.expected.emplace_back(Shape{1, 1, 2, 3}, element::f32, std::vector<float>{0, 2, 4, 1, 3, 5});
    res.expected.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 1, 2, 3});
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
        mean_scale_dynamic_layout(),
        resize_to_network_height(),
        resize_to_network_width(),
        resize_from_spatial_dims(),
        resize_i8(),
        resize_to_network_width_height(),
        resize_to_specified_width_height(),
        resize_lvalues(),
        convert_layout_nhwc_to_nchw_lvalue(),
        convert_layout_nhwc_to_net_no_tensor_shape(),
        resize_and_convert_layout(),
        convert_color_nv12_to_bgr_two_planes(),
        convert_color_nv12_single_plane(),
        convert_color_nv12_layout_resize(),
        element_type_before_convert_color_nv12(),
        postprocess_2_inputs_basic(),
        pre_and_post_processing()
             };
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs, ReferencePreprocessTest,
                         ::testing::ValuesIn(allPreprocessTests()),
                         ReferencePreprocessTest::getTestCaseName);
