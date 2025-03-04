// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using std::make_shared;
using testing::UnorderedElementsAre;

TEST(type_prop, result) {
    const auto arg_shape = Shape{1, 2, 3, 4, 5};
    auto arg = make_shared<ov::op::v0::Constant>(element::f32, arg_shape);

    auto result = make_shared<ov::op::v0::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_EQ(result->get_output_shape(0), arg_shape);
}

TEST(type_prop, result_dynamic_shape) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto result = make_shared<ov::op::v0::Result>(arg);

    EXPECT_EQ(result->get_output_element_type(0), element::f32);
    EXPECT_TRUE(result->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, result_layout) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    result->set_layout("NHWC");
    EXPECT_EQ(result->get_layout(), "NHWC");
    result->set_layout(ov::Layout());
    EXPECT_TRUE(result->get_layout().empty());
    EXPECT_EQ(result->output(0).get_rt_info().count(ov::LayoutAttribute::get_type_info_static()), 0);
}

TEST(type_prop, result_layout_empty) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    EXPECT_TRUE(result->get_layout().empty());
}

TEST(type_prop, result_layout_invalid) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto result = make_shared<ov::op::v0::Result>(a);
    result->output(0).get_rt_info()[ov::LayoutAttribute::get_type_info_static()] = "NCHW";  // incorrect way
    ASSERT_THROW(result->get_layout(), ov::Exception);
}

using TypePropResultV0Test = TypePropOpTest<op::v0::Result>;

TEST_F(TypePropResultV0Test, set_specific_output_name_by_output) {
    auto a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).set_names({"input"});

    auto result = make_op(a);

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("input"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("input"));

    result->output(0).set_names({"out"});
    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input", "out"));
    EXPECT_THAT(a->get_output_tensor(0).get_names(), UnorderedElementsAre("input", "out"));
}

TEST_F(TypePropResultV0Test, set_specific_output_name_by_tensor_desc) {
    auto a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).set_names({"input"});

    auto result = make_op(a);

    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("input"));

    result->get_output_tensor(0).set_names({"out"});
    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input", "out"));
    EXPECT_THAT(a->get_output_tensor(0).get_names(), UnorderedElementsAre("input", "out"));
}

TEST_F(TypePropResultV0Test, change_specific_output_name) {
    auto a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).set_names({"input"});

    auto result = make_op(a);

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("input"));

    result->get_output_tensor(0).set_names({"out"});

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input", "out"));
    EXPECT_THAT(a->get_output_tensor(0).get_names(), UnorderedElementsAre("input", "out"));

    result->output(0).set_names({"new output"});

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("new output"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("new output"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input", "new output"));
    EXPECT_THAT(a->get_output_tensor(0).get_names(), UnorderedElementsAre("input", "new output"));
}

TEST_F(TypePropResultV0Test, add_specific_output_name) {
    auto a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).set_names({"input"});

    auto result = make_op(a);

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("input"));

    result->output(0).set_names({"out"});
    result->get_output_tensor(0).add_names({"extra output name", "o1"});
    result->output(0).add_names({"extra output name", "o2"});

    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out", "extra output name", "o1", "o2"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("out", "extra output name", "o1", "o2"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input", "out", "extra output name", "o1", "o2"));
    EXPECT_THAT(a->get_output_tensor(0).get_names(),
                UnorderedElementsAre("input", "out", "extra output name", "o1", "o2"));
}

TEST_F(TypePropResultV0Test, preserve_specific_name_on_input_replace) {
    const auto a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).set_names({"input a"});

    const auto result = make_op(a, true);
    result->output(0).set_names({"out"});

    EXPECT_THAT(result->input(0).get_tensor().get_names(), UnorderedElementsAre("out", "input a"));
    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out"));

    const auto b = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    b->get_output_tensor(0).set_names({"input b"});

    result->input(0).replace_source_output(b);
    result->validate_and_infer_types();

    EXPECT_THAT(result->input(0).get_tensor().get_names(), UnorderedElementsAre("input b", "out"));
    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("out"));
    EXPECT_THAT(a->output(0).get_names(), UnorderedElementsAre("input a"));
}

TEST_F(TypePropResultV0Test, take_input_node_names) {
    const auto c = std::make_shared<Constant>(element::f32, Shape{2}, std::vector<float>{2.f, 1.f});
    c->get_output_tensor(0).set_names({"constant data"});
    const auto result = make_op(c, true);

    EXPECT_THAT(result->input(0).get_tensor().get_names(), UnorderedElementsAre("constant data"));
    EXPECT_THAT(result->output(0).get_names(), UnorderedElementsAre("constant data"));

    const auto new_const = std::make_shared<Constant>(element::f32, Shape{2}, std::vector<float>{0.f, 0.f});

    result->input(0).replace_source_output(new_const);
    result->validate_and_infer_types();

    EXPECT_THAT(c->get_output_tensor(0).get_names(), testing::IsEmpty());
    EXPECT_THAT(result->get_input_tensor(0).get_names(), UnorderedElementsAre("constant data"));
    EXPECT_THAT(result->get_output_tensor(0).get_names(), UnorderedElementsAre("constant data"));
}
}  // namespace test
}  // namespace ov
