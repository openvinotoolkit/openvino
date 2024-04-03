// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace reference_tests;
namespace {

struct RefPreprocessParams {
    RefPreprocessParams(const std::string& val) : name(val) {}
    std::function<std::shared_ptr<ov::Model>()> function;
    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected;
    float abs_threshold = 0.01f;
    float rel_threshold = 0.01f;
    std::string name;
};

class ReferencePreprocessTest : public testing::TestWithParam<RefPreprocessParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        legacy_compare = true;
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
}  // namespace

static std::shared_ptr<Model> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto c = op::v0::Constant::create(type, {1}, {0});
    auto op = std::make_shared<op::v1::Add>(data1, c);
    op->set_friendly_name("Add0");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    return std::make_shared<ov::Model>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Model> create_n_inputs(const int N, const element::Type& type, const PartialShape& shape) {
    auto params = ParameterVector();
    auto results = ResultVector();
    for (int i = 1; i <= N; i++) {
        auto param = std::make_shared<op::v0::Parameter>(type, shape);
        param->set_friendly_name("input" + std::to_string(i));
        param->get_output_tensor(0).set_names({"tensor_input" + std::to_string(i)});
        auto c1 = op::v0::Constant::create(type, {1}, {0});
        auto op1 = std::make_shared<op::v1::Add>(param, c1);
        op1->set_friendly_name("Add" + std::to_string(i));
        auto res1 = std::make_shared<op::v0::Result>(op1);
        res1->set_friendly_name("Result" + std::to_string(i));
        res1->get_output_tensor(0).set_names({"tensor_output" + std::to_string(i)});
        results.push_back(res1);
        params.push_back(param);
    }
    return std::make_shared<ov::Model>(results, params);
}

static RefPreprocessParams simple_mean_scale() {
    RefPreprocessParams res("simple_mean_scale");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().preprocess().mean(1.f).scale(2.f);
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 2, 2},
                            element::f32,
                            std::vector<float>{1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23.});
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});
    return res;
}

static RefPreprocessParams scale_then_mean() {
    RefPreprocessParams res("scale_then_mean");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().preprocess().scale(2.0f).mean(2.0f);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 2, 2},
                            element::f32,
                            std::vector<float>{2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 100., 200.});
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{-1., 0, 1., 2., 3., 4., 5., 6., 7., 8., 48., 98.});
    return res;
}

static RefPreprocessParams convert_only() {
    RefPreprocessParams res("convert_only");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{1, 1, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_element_type(element::i16);
        p.input()
            .preprocess()
            .convert_element_type(element::f32)
            .scale(3.f)
            .convert_element_type(element::u8)
            .convert_element_type(element::f32);
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_element_type(element::i16);
        p.input().preprocess().convert_element_type(element::f32).scale(2.f).convert_element_type(element::u8);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 2, 2},
                            element::i16,
                            std::vector<int16_t>{2, 3, 6, 8, 10, 12, 14, 16, 18, 20, 10000, 200});
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1, 1, 3, 4, 5, 6, 7, 8, 9, 10, (uint8_t)5000, 100});
    return res;
}

static RefPreprocessParams tensor_element_type_and_scale() {
    RefPreprocessParams res("tensor_element_type_and_scale");
    res.function = []() {
        auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_element_type(element::f32);
        p.input().preprocess().scale(2.0f).convert_element_type(element::i8);
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().preprocess().custom([](const Output<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
            return abs;
        });
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::i32, std::vector<int32_t>{0, 4, -6});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i32, std::vector<int32_t>{0, 4, 6});
    return res;
}

static RefPreprocessParams test_multiple() {
    RefPreprocessParams res("test_multiple");
    res.function = []() {
        auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
        auto p = PrePostProcessor(f);
        auto p1 = std::move(p);
        p1.input().tensor().set_element_type(element::f32).set_layout("?CHW");
        p1.input().preprocess().mean(1.f);
        p1.input().preprocess().scale(2.f);
        p1.input().preprocess().mean({1.f, 2.f, 3.f});
        p1.input().preprocess().scale({2.f, 3.f, 4.f});
        p1.input().preprocess().custom([](const Output<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
            return abs;
        });
        p1.input().preprocess().convert_element_type(element::i8);
        f = p1.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{-9., 17., -1.});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i8, std::vector<int8_t>{3, 2, 1});
    return res;
}

static RefPreprocessParams test_2_inputs_basic() {
    RefPreprocessParams res("test_2_inputs_basic");
    res.function = []() {
        auto f = create_n_inputs(2, element::f32, Shape{1, 3, 1, 1});
        auto p = PrePostProcessor(f);
        p.input(0).preprocess().mean(1.f);
        p.input("tensor_input2").preprocess().mean(1.f).scale(2.0f);
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NC??");
        p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f});
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("N...C");
        p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f});
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_dynamic_shape();
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
        p.input().model().set_layout("NHWC");
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_dynamic_shape();
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(element::f32,
                            Shape{1, 1, 2, 6},
                            std::vector<float>{0., 1., 2., 3., 4., 5., 0., 1., 2., 3., 4., 5.});
    res.expected.emplace_back(Shape{1, 1, 2, 2}, element::f32, std::vector<float>{1., 4., 1., 4.});
    return res;
}

static RefPreprocessParams resize_from_spatial_dims() {
    RefPreprocessParams res("resize_from_spatial_dims");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 1, 1});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_static_shape(1, 4);
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
        p.input().model().set_layout("NCHW");
        p.build();
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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_dynamic_shape();
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(element::i8, Shape{1, 3, 2, 2}, std::vector<int8_t>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::i8, std::vector<int8_t>{0, 1, 2});
    return res;
}

static RefPreprocessParams resize_to_network_width_height() {
    RefPreprocessParams res("resize_to_network_width_height");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 1, 4, 4});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_static_shape(5, 5);
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST);
        p.input().model().set_layout("...HW");
        p.build();
        return f;
    };

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
        auto p = PrePostProcessor(f);
        p.input().tensor().set_spatial_dynamic_shape();
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST, 4, 4);
        p.input().model().set_layout("...HW");
        p.build();
        return f;
    };

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

static RefPreprocessParams convert_layout_nhwc_to_nchw() {
    RefPreprocessParams res("convert_layout_nhwc_to_nchw");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 3, 2, 2});
        f->get_parameters()[0]->set_layout("NCHW");

        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NHWC");
        p.input().preprocess().convert_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::u8,
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1,
                                                   4,
                                                   7,
                                                   10,  // R
                                                   2,
                                                   5,
                                                   8,
                                                   11,  // G
                                                   3,
                                                   6,
                                                   9,
                                                   12});  // B
    return res;
}

static RefPreprocessParams convert_layout_nhwc_to_nchw_fully_dynamic() {
    RefPreprocessParams res("convert_layout_nhwc_to_nchw_fully_dynamic");
    res.function = []() {
        auto f = create_simple_function(element::u8, PartialShape::dynamic());
        f->get_parameters()[0]->set_layout("NCHW");

        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NHWC");
        p.input().preprocess().convert_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(element::u8,
                            Shape{1, 2, 2, 3},
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1,
                                                   4,
                                                   7,
                                                   10,  // R
                                                   2,
                                                   5,
                                                   8,
                                                   11,  // G
                                                   3,
                                                   6,
                                                   9,
                                                   12});  // B
    return res;
}

static RefPreprocessParams convert_layout_hwc_to_nchw() {
    RefPreprocessParams res("convert_layout_hwc_to_nchw");
    res.function = []() {
        auto f = create_simple_function(element::f32, {Dimension::dynamic(), 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("HWC").set_element_type(element::u8);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{2, 2, 3},
                            element::u8,
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{1,
                                                 4,
                                                 7,
                                                 10,  // R
                                                 2,
                                                 5,
                                                 8,
                                                 11,  // G
                                                 3,
                                                 6,
                                                 9,
                                                 12});  // B
    return res;
}

static RefPreprocessParams convert_layout_hwc_to_nchw_fully_dynamic() {
    RefPreprocessParams res("convert_layout_hwc_to_nchw_fully_dynamic");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape::dynamic());
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("HWC").set_element_type(element::u8);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };
    res.inputs.emplace_back(element::u8,
                            Shape{2, 2, 3},
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{1,
                                                 4,
                                                 7,
                                                 10,  // R
                                                 2,
                                                 5,
                                                 8,
                                                 11,  // G
                                                 3,
                                                 6,
                                                 9,
                                                 12});  // B
    return res;
}

static RefPreprocessParams convert_layout_nhwc_to_net_no_tensor_shape() {
    RefPreprocessParams res("convert_layout_nhwc_to_net_no_tensor_shape");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 3, 2, 2});
        f->get_parameters()[0]->set_layout("NCHW");
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NHWC");
        p.input().preprocess().convert_layout();
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::u8,
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1,
                                                   4,
                                                   7,
                                                   10,  // R
                                                   2,
                                                   5,
                                                   8,
                                                   11,  // G
                                                   3,
                                                   6,
                                                   9,
                                                   12});  // B
    return res;
}

static RefPreprocessParams convert_layout_by_dims() {
    RefPreprocessParams res("convert_layout_by_dims");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().preprocess().convert_layout({0, 3, 1, 2});
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::u8,
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1,
                                                   4,
                                                   7,
                                                   10,  // R
                                                   2,
                                                   5,
                                                   8,
                                                   11,  // G
                                                   3,
                                                   6,
                                                   9,
                                                   12});  // B
    return res;
}

static RefPreprocessParams convert_layout_by_dims_multi() {
    RefPreprocessParams res("convert_layout_by_dims_multi");
    res.function = []() {
        auto f = create_simple_function(element::f32, {1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input()
            .preprocess()
            .convert_layout({0, 1, 3, 2})   // NHWC->NHCW
            .convert_layout({0, 2, 1, 3});  // NHCW->NCHW
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::f32,
                            std::vector<float>{1,
                                               2,
                                               3,  // [H=0, W=0]
                                               4,
                                               5,
                                               6,  // [H=0, W=1]
                                               7,
                                               8,
                                               9,  // [H=1, W=0]
                                               10,
                                               11,
                                               12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{1,
                                                 4,
                                                 7,
                                                 10,  // R
                                                 2,
                                                 5,
                                                 8,
                                                 11,  // G
                                                 3,
                                                 6,
                                                 9,
                                                 12});  // B
    return res;
}

static RefPreprocessParams convert_layout_by_dims_multi_layout() {
    RefPreprocessParams res("convert_layout_by_dims_multi_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, {1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("N??C");
        p.input()
            .preprocess()
            .convert_layout({0, 1, 3, 2})   // NHWC->NHCW
            .mean({1, 2, 2})                // Apply means to 'C' channel
            .convert_layout({0, 2, 1, 3});  // NHCW->NCHW
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::f32,
                            std::vector<float>{1,
                                               2,
                                               3,  // [H=0, W=0, RGB]
                                               4,
                                               5,
                                               6,  // [H=0, W=1]
                                               7,
                                               8,
                                               9,  // [H=1, W=0]
                                               10,
                                               11,
                                               12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{1 - 1,
                                                 4 - 1,
                                                 7 - 1,
                                                 10 - 1,  // R
                                                 2 - 2,
                                                 5 - 2,
                                                 8 - 2,
                                                 11 - 2,  // G
                                                 3 - 2,
                                                 6 - 2,
                                                 9 - 2,
                                                 12 - 2});  // B
    return res;
}

static RefPreprocessParams resize_and_convert_layout() {
    RefPreprocessParams res("resize_and_convert_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW").set_spatial_dynamic_shape();
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR).convert_layout();
        p.input().model().set_layout("NHWC");
        p.build();
        return f;
    };

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
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::u8, PartialShape{1, 4, 4, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
        p.input().preprocess().convert_color(ColorFormat::BGR);
        p.build();
        return f;
    };

    // clang-format off
    auto input_y = std::vector<uint8_t> {81, 81, 145, 145,      // RRGG
                                         81, 81, 145, 145,      // RRGG
                                         41, 41, 81, 81,        // BBRR
                                         41, 41, 81, 81};       // BBRR
    auto input_shape_y = Shape{1, 4, 4, 1};
    auto input_uv = std::vector<uint8_t> {90, 240,      // R (2x2)
                                          54, 34,       // G (2x2)
                                          240, 110,     // B (2x2)
                                          90, 240};     // R (2x2)
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
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
        p.input().preprocess().convert_color(ColorFormat::RGB);
        p.build();
        return f;
    };

    // clang-format off
    auto input = std::vector<float> {  81, 81, 145, 145,      // RRGG
                                       81, 81, 145, 145,      // RRGG
                                       41, 41, 81, 81,        // BBRR
                                       41, 41, 81, 81,        // BBRR
                                       90, 240, 54, 34, 240, 110, 90, 240};     // UV (RGBR)
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
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 3, 2, 2});
        auto p = PrePostProcessor(f);
        p.input()
            .tensor()
            .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
            .set_element_type(element::u8)
            .set_spatial_dynamic_shape();
        p.input()
            .preprocess()
            .convert_color(ColorFormat::RGB)
            .convert_layout()
            .convert_element_type(element::f32)
            .resize(ResizeAlgorithm::RESIZE_NEAREST);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };

    // clang-format off
    auto input = std::vector<uint8_t> {81, 81, 145, 145,      // RRGG
                                       81, 81, 145, 145,      // RRGG
                                       41, 41, 81, 81,        // BBRR
                                       41, 41, 81, 81,        // BBRR
                                       90, 240, 54, 34, 240, 110, 90, 240};     // UV (RGBR)
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
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_element_type(element::u8).set_color_format(ColorFormat::NV12_TWO_PLANES);
        p.input().preprocess().convert_element_type(element::f32).convert_color(ColorFormat::RGB);
        p.input().model().set_layout("NHWC");
        p.build();
        return f;
    };

    // clang-format off
    auto input_y = std::vector<uint8_t> {81, 81, 81, 81};
    auto input_shape_y = Shape{1, 2, 2, 1};
    auto input_uv = std::vector<uint8_t> {90, 240};
    auto input_shape_uv = Shape{1, 1, 1, 2};
    auto exp_out = std::vector<float> {255, 0, 0,  255, 0, 0,  255, 0, 0,  255, 0,  0};
    auto out_shape = Shape{1, 2, 2, 3};
    // clang-format on
    res.inputs.emplace_back(element::u8, input_shape_y, input_y);
    res.inputs.emplace_back(element::u8, input_shape_uv, input_uv);
    res.expected.emplace_back(out_shape, element::f32, exp_out);
    return res;
}

static RefPreprocessParams convert_color_i420_to_bgr_three_planes() {
    RefPreprocessParams res("convert_color_i420_to_bgr_three_planes");
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::u8, PartialShape{1, 4, 4, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
        p.input().preprocess().convert_color(ColorFormat::BGR);
        return p.build();
    };

    // clang-format off
    auto input_y = std::vector<uint8_t> {81, 81, 145, 145,      // RRGG
                                         81, 81, 145, 145,      // RRGG
                                         41, 41, 81, 81,        // BBRR
                                         41, 41, 81, 81};       // BBRR
    auto input_shape_y = Shape{1, 4, 4, 1};
    auto input_u = std::vector<uint8_t> {90,      // R (2x2)
                                         54,       // G (2x2)
                                         240,     // B (2x2)
                                         90};     // R (2x2)
    auto input_v = std::vector<uint8_t> {240,      // R (2x2)
                                         34,       // G (2x2)
                                         110,     // B (2x2)
                                         240};     // R (2x2)
    auto input_shape_uv = Shape{1, 2, 2, 1};
    auto exp_out = std::vector<uint8_t> {0, 0, 255,  0, 0, 255,  0, 255, 0,  0, 255, 0,
                                         0, 0, 255,  0, 0, 255,  0, 255, 0,  0, 255, 0,
                                         255, 0, 0,  255, 0, 0,  0, 0, 255,  0, 0, 255,
                                         255, 0, 0,  255, 0, 0,  0, 0, 255,  0, 0, 255};
    auto out_shape = Shape{1, 4, 4, 3};
    // clang-format on
    res.inputs.emplace_back(element::u8, input_shape_y, input_y);
    res.inputs.emplace_back(element::u8, input_shape_uv, input_u);
    res.inputs.emplace_back(element::u8, input_shape_uv, input_v);
    res.expected.emplace_back(out_shape, element::u8, exp_out);
    return res;
}

static RefPreprocessParams convert_color_i420_single_plane() {
    RefPreprocessParams res("convert_color_i420_single_plane");
    res.abs_threshold = 1.f;  // Allow small color conversion deviations
    res.rel_threshold = 1.f;  // Ignore relative pixel values comparison (100%)
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
        p.input().preprocess().convert_color(ColorFormat::RGB);
        return p.build();
    };

    // clang-format off
    auto input = std::vector<float> {  81, 81, 145, 145,      // RRGG
                                       81, 81, 145, 145,      // RRGG
                                       41, 41, 81, 81,        // BBRR
                                       41, 41, 81, 81,        // BBRR
                                       90, 54, 240, 90, 240, 34, 110, 240};     // UV (RGBR)
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

static RefPreprocessParams set_shape_custom_crop() {
    RefPreprocessParams res("set_shape_custom_crop");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{2, 2, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_shape({-1, -1, -1, -1});
        p.input().preprocess().custom([](const Output<Node>& node) {
            // Add custom crop to model's dimensions using 'Slice' operation
            // Middle part 2x2x2x2 of original user's 4x4x4x4 input tensor will be extracted
            auto start = ov::op::v0::Constant::create(element::i32, {4}, {1, 1, 1, 1});
            auto stop = ov::op::v0::Constant::create(element::i32, {4}, {3, 3, 3, 3});
            auto step = ov::op::v0::Constant::create(element::i32, {4}, {1, 1, 1, 1});
            auto axis = ov::op::v0::Constant::create(element::i32, {4}, {0, 1, 2, 3});
            auto slice = std::make_shared<ov::op::v8::Slice>(node, start, stop, step, axis);
            return slice;
        });
        p.build();
        return f;
    };
    auto input_shape = Shape{4, 4, 4, 4};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    res.inputs.emplace_back(element::f32, input_shape, input_values);
    res.expected.emplace_back(
        Shape{2, 2, 2, 2},
        element::f32,
        std::vector<float>{85, 86, 89, 90, 101, 102, 105, 106, 149, 150, 153, 154, 165, 166, 169, 170});
    return res;
}

static RefPreprocessParams set_shape_with_resize() {
    RefPreprocessParams res("set_shape_with_resize");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 3, 1, 1});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_shape({1, 2, 2, 3}).set_layout("NHWC");
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
        p.input().model().set_layout("NCHW");
        p.build();
        return f;
    };
    auto input_size = 1 * 2 * 2 * 3;
    std::vector<float> input_values(input_size);
    std::iota(input_values.begin(), input_values.end(), 0.f);
    res.inputs.emplace_back(element::f32, Shape{1, 2, 2, 3}, std::vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{1, 2, 3});
    return res;
}

static RefPreprocessParams preprocess_crop_basic() {
    RefPreprocessParams res("preprocess_crop_basic");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{2, 2, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_shape({4, 4, 4, 4});
        p.input().preprocess().crop({1, 1, 1, 1}, {-1, -1, -1, -1});
        p.build();
        return f;
    };
    auto input_shape = Shape{4, 4, 4, 4};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    res.inputs.emplace_back(element::f32, input_shape, input_values);
    res.expected.emplace_back(
        Shape{2, 2, 2, 2},
        element::f32,
        std::vector<float>{85, 86, 89, 90, 101, 102, 105, 106, 149, 150, 153, 154, 165, 166, 169, 170});
    return res;
}

static RefPreprocessParams preprocess_crop_2axis_dynamic() {
    RefPreprocessParams res("preprocess_crop_2axis_dynamic");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape::dynamic());
        auto p = PrePostProcessor(f);
        auto max_int = std::numeric_limits<int>::max();
        p.input().preprocess().crop({0, 0, 1, 1}, {max_int, max_int, max_int, max_int});
        p.build();
        return f;
    };
    auto input_shape = Shape{1, 3, 2, 2};
    std::vector<float> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), 0.f);
    res.inputs.emplace_back(element::f32, Shape{1, 3, 2, 2}, input_values);
    res.expected.emplace_back(Shape{1, 3, 1, 1}, element::f32, std::vector<float>{3, 7, 11});
    return res;
}

static RefPreprocessParams postprocess_2_inputs_basic() {
    RefPreprocessParams res("postprocess_2_inputs_basic");
    res.function = []() {
        auto f = create_n_inputs(2, element::f32, Shape{1, 3, 1, 2});
        auto p = PrePostProcessor(f);
        p.output("tensor_output1").model().set_layout("NCHW");
        p.output("tensor_output1").postprocess().convert_layout();
        p.output("tensor_output1").tensor().set_layout("NHWC");
        p.output("tensor_output2").postprocess().convert_element_type();
        p.output("tensor_output2").tensor().set_element_type(element::u8);
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{1.1, 2.1, 3.1, 4.1, 5.1, 6.1});
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{1.1, 2.1, 3.1, 4.1, 5.1, 6.1});
    res.expected.emplace_back(Shape{1, 1, 2, 3}, element::f32, std::vector<float>{1.1, 3.1, 5.1, 2.1, 4.1, 6.1});
    res.expected.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    return res;
}

static RefPreprocessParams post_convert_layout_by_dims() {
    RefPreprocessParams res("post_convert_layout_by_dims");
    res.function = []() {
        auto f = create_simple_function(element::u8, {1, 2, 2, 3});
        auto p = PrePostProcessor(f);
        p.output().postprocess().convert_layout({0, 3, 1, 2});
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::u8,
                            std::vector<uint8_t>{1,
                                                 2,
                                                 3,  // [H=0, W=0, RGB]
                                                 4,
                                                 5,
                                                 6,  // [H=0, W=1]
                                                 7,
                                                 8,
                                                 9,  // [H=1, W=0]
                                                 10,
                                                 11,
                                                 12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::u8,
                              std::vector<uint8_t>{1,
                                                   4,
                                                   7,
                                                   10,  // R
                                                   2,
                                                   5,
                                                   8,
                                                   11,  // G
                                                   3,
                                                   6,
                                                   9,
                                                   12});  // B
    return res;
}

static RefPreprocessParams post_convert_layout_by_dims_multi() {
    RefPreprocessParams res("post_convert_layout_by_dims_multi");
    res.function = []() {
        auto f = create_simple_function(element::f32, {1, 2, 2, 3});
        auto p = PrePostProcessor(f);
        p.output().postprocess().convert_layout({0, 1, 3, 2});  // NHWC->NHCW;
        p.output().postprocess().convert_layout({0, 2, 1, 3});  // NHCW->NCHW;
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 2, 2, 3},
                            element::f32,
                            std::vector<float>{1,
                                               2,
                                               3,  // [H=0, W=0]
                                               4,
                                               5,
                                               6,  // [H=0, W=1]
                                               7,
                                               8,
                                               9,  // [H=1, W=0]
                                               10,
                                               11,
                                               12});  // [H=1, W=1]
    res.expected.emplace_back(Shape{1, 3, 2, 2},
                              element::f32,
                              std::vector<float>{1,
                                                 4,
                                                 7,
                                                 10,  // R
                                                 2,
                                                 5,
                                                 8,
                                                 11,  // G
                                                 3,
                                                 6,
                                                 9,
                                                 12});  // B
    return res;
}

static RefPreprocessParams post_convert_color_rgb_to_bgr() {
    RefPreprocessParams res("post_convert_color_rgb_to_bgr");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{2, 1, 1, 3});
        auto p = PrePostProcessor(f);
        p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
        p.output().postprocess().convert_color(ColorFormat::BGR);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6});
    res.expected.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{3, 2, 1, 6, 5, 4});
    return res;
}

static RefPreprocessParams post_convert_color_bgr_to_rgb() {
    RefPreprocessParams res("post_convert_color_bgr_to_rgb");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{2, 1, 1, 3});
        auto p = PrePostProcessor(f);
        p.output().model().set_layout("NHWC").set_color_format(ColorFormat::BGR);
        p.output().postprocess().convert_color(ColorFormat::RGB);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6});
    res.expected.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{3, 2, 1, 6, 5, 4});
    return res;
}

static RefPreprocessParams pre_and_post_processing() {
    RefPreprocessParams res("pre_and_post_processing");
    res.function = []() {
        auto f = create_n_inputs(2, element::f32, Shape{1, 3, 1, 2});
        auto p = PrePostProcessor(f);
        p.input(0).tensor().set_element_type(element::u8);
        p.input(0).preprocess().convert_element_type(element::f32).mean(1.f);
        p.input(1).preprocess().scale(2.f);
        p.output("tensor_output1").model().set_layout("NCHW");
        p.output("tensor_output1").postprocess().convert_layout();
        p.output("tensor_output1").tensor().set_layout("NHWC");
        p.output("tensor_output2").postprocess().convert_element_type();
        p.output("tensor_output2").tensor().set_element_type(element::u8);
        p.build();
        return f;
    };
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    res.inputs.emplace_back(Shape{1, 3, 1, 2}, element::f32, std::vector<float>{2.2, 4.2, 6.2, 2.4, 4.4, 6.4});
    res.expected.emplace_back(Shape{1, 1, 2, 3}, element::f32, std::vector<float>{0, 2, 4, 1, 3, 5});
    res.expected.emplace_back(Shape{1, 3, 1, 2}, element::u8, std::vector<uint8_t>{1, 2, 3, 1, 2, 3});
    return res;
}

static RefPreprocessParams rgb_to_bgr() {
    RefPreprocessParams res("rgb_to_bgr");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{2, 1, 1, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::RGB);
        p.input().preprocess().convert_color(ColorFormat::BGR);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6});
    res.expected.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{3, 2, 1, 6, 5, 4});
    return res;
}

static RefPreprocessParams bgr_to_rgb() {
    RefPreprocessParams res("bgr_to_rgb");
    res.function = []() {
        auto f = create_simple_function(element::f32, Shape{2, 1, 1, 3});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::BGR);
        p.input().preprocess().convert_color(ColorFormat::RGB);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6});
    res.expected.emplace_back(Shape{2, 3, 1, 1}, element::f32, std::vector<float>{3, 2, 1, 6, 5, 4});
    return res;
}

static RefPreprocessParams reverse_channels_nchw() {
    RefPreprocessParams res("reverse_channels_nchw");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        p.input().preprocess().reverse_channels();
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 2, 2, 2}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    res.expected.emplace_back(Shape{1, 2, 2, 2}, element::f32, std::vector<float>{5, 6, 7, 8, 1, 2, 3, 4});
    return res;
}

static RefPreprocessParams color_cut_last_channel() {
    RefPreprocessParams res("color_cut_last_channel");
    auto input_tensor = reference_tests::Tensor(Shape{1, 2, 2, 4},
                                                element::f32,
                                                std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 6, 7, 8, 9});
    auto exp_3_channels = reference_tests::Tensor(Shape{1, 2, 2, 3},
                                                  element::f32,
                                                  std::vector<float>{1, 2, 3, 5, 6, 7, 3, 4, 5, 6, 7, 8});
    auto inv_3_channels = reference_tests::Tensor(Shape{1, 2, 2, 3},
                                                  element::f32,
                                                  std::vector<float>{3, 2, 1, 7, 6, 5, 5, 4, 3, 8, 7, 6});
    res.function = []() {
        auto f = create_n_inputs(4, element::f32, Shape{1, 2, 2, 3});
        auto prep = PrePostProcessor(f);
        prep.input(0).tensor().set_color_format(ColorFormat::RGBX);
        prep.input(0).preprocess().convert_color(ColorFormat::RGB);

        prep.input(1).tensor().set_color_format(ColorFormat::RGBX);
        prep.input(1).preprocess().convert_color(ColorFormat::BGR);

        prep.input(2).tensor().set_color_format(ColorFormat::BGRX);
        prep.input(2).preprocess().convert_color(ColorFormat::BGR);

        prep.input(3).tensor().set_color_format(ColorFormat::BGRX);
        prep.input(3).preprocess().convert_color(ColorFormat::RGB);
        return prep.build();
    };

    res.inputs = std::vector<reference_tests::Tensor>{input_tensor, input_tensor, input_tensor, input_tensor};
    res.expected = std::vector<reference_tests::Tensor>{exp_3_channels, inv_3_channels, exp_3_channels, inv_3_channels};
    return res;
}

static RefPreprocessParams reverse_channels_dyn_layout() {
    RefPreprocessParams res("reverse_channels_dyn_layout");
    res.function = []() {
        auto f = create_simple_function(element::f32, PartialShape{1, 1, 3, 2});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_color_format(ColorFormat::BGR).set_layout("...CN");
        p.input().preprocess().convert_color(ColorFormat::RGB);
        p.build();
        return f;
    };

    res.inputs.emplace_back(Shape{1, 1, 3, 2}, element::f32, std::vector<float>{1, 2, 3, 4, 5, 6});
    res.expected.emplace_back(Shape{1, 1, 3, 2}, element::f32, std::vector<float>{5, 6, 3, 4, 1, 2});
    return res;
}

static RefPreprocessParams reverse_dyn_shape() {
    RefPreprocessParams res("reverse_dyn_shape");
    res.function = []() {
        auto f = create_simple_function(
            element::u8,
            PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        p.input().preprocess().reverse_channels();
        p.build();
        return f;
    };

    res.inputs.emplace_back(element::u8,
                            Shape{2, 2, 1, 3},
                            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    res.expected.emplace_back(Shape{2, 2, 1, 3},
                              element::u8,
                              std::vector<uint8_t>{4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9});
    return res;
}

static RefPreprocessParams reverse_dyn_channels() {
    RefPreprocessParams res("reverse_dyn_channels");
    res.function = []() {
        auto f =
            create_simple_function(element::u8,
                                   PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic()});
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        p.input().preprocess().reverse_channels();
        p.build();
        return f;
    };

    res.inputs.emplace_back(element::u8,
                            Shape{2, 2, 1, 3},
                            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    res.expected.emplace_back(Shape{2, 2, 1, 3},
                              element::u8,
                              std::vector<uint8_t>{4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9});
    return res;
}

static RefPreprocessParams reverse_fully_dyn_shape() {
    RefPreprocessParams res("reverse_fully_dyn_shape");
    res.function = []() {
        auto f = create_simple_function(element::u8, PartialShape::dynamic());
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("...C??");
        p.input().preprocess().reverse_channels();
        p.build();
        return f;
    };

    res.inputs.emplace_back(element::u8,
                            Shape{2, 2, 1, 3},
                            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    res.expected.emplace_back(Shape{2, 2, 1, 3},
                              element::u8,
                              std::vector<uint8_t>{4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9});
    return res;
}

std::vector<RefPreprocessParams> allPreprocessTests() {
    return std::vector<RefPreprocessParams>{simple_mean_scale(),
                                            scale_then_mean(),
                                            convert_only(),
                                            convert_element_type_and_scale(),
                                            tensor_element_type_and_scale(),
                                            custom_preprocessing(),
                                            test_multiple(),
                                            test_2_inputs_basic(),
                                            mean_scale_vector_tensor_layout(),
                                            mean_scale_dynamic_layout(),
                                            resize_to_network_height(),
                                            resize_to_network_width(),
                                            resize_from_spatial_dims(),
                                            resize_i8(),
                                            resize_to_network_width_height(),
                                            resize_to_specified_width_height(),
                                            convert_layout_nhwc_to_nchw(),
                                            convert_layout_nhwc_to_nchw_fully_dynamic(),
                                            convert_layout_nhwc_to_net_no_tensor_shape(),
                                            convert_layout_by_dims(),
                                            convert_layout_by_dims_multi(),
                                            convert_layout_by_dims_multi_layout(),
                                            convert_layout_hwc_to_nchw(),
                                            convert_layout_hwc_to_nchw_fully_dynamic(),
                                            resize_and_convert_layout(),
                                            convert_color_nv12_to_bgr_two_planes(),
                                            convert_color_nv12_single_plane(),
                                            convert_color_nv12_layout_resize(),
                                            element_type_before_convert_color_nv12(),
                                            convert_color_i420_to_bgr_three_planes(),
                                            convert_color_i420_single_plane(),
                                            preprocess_crop_basic(),
                                            preprocess_crop_2axis_dynamic(),
                                            set_shape_custom_crop(),
                                            set_shape_with_resize(),
                                            postprocess_2_inputs_basic(),
                                            post_convert_layout_by_dims(),
                                            post_convert_layout_by_dims_multi(),
                                            post_convert_color_rgb_to_bgr(),
                                            post_convert_color_bgr_to_rgb(),
                                            pre_and_post_processing(),
                                            rgb_to_bgr(),
                                            bgr_to_rgb(),
                                            color_cut_last_channel(),
                                            reverse_channels_nchw(),
                                            reverse_channels_dyn_layout(),
                                            reverse_dyn_shape(),
                                            reverse_dyn_channels(),
                                            reverse_fully_dyn_shape()};
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs,
                         ReferencePreprocessTest,
                         ::testing::ValuesIn(allPreprocessTests()),
                         ReferencePreprocessTest::getTestCaseName);
