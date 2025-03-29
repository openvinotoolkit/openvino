// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov;
using namespace ov::opset11;
using namespace testing;

template <class TOp>
class PriorBoxTest : public TypePropOpTest<TOp> {
protected:
    using Attrs = typename TOp::Attributes;

    void SetUp() override {
        this->attrs.min_size = {2.0f, 3.0f};
        this->attrs.aspect_ratio = {1.5f, 2.0f, 2.5f};
        this->attrs.scale_all_sizes = false;
    }

    template <class T = TOp, typename std::enable_if<std::is_same<T, op::v0::PriorBox>::value>::type* = nullptr>
    static void check_def_attrs(const TOp* const op) {
        const auto& attrs = op->get_attrs();

        EXPECT_EQ(attrs.min_size, std::vector<float>());
        EXPECT_EQ(attrs.max_size, std::vector<float>());
        EXPECT_EQ(attrs.aspect_ratio, std::vector<float>());
        EXPECT_EQ(attrs.density, std::vector<float>());
        EXPECT_EQ(attrs.fixed_ratio, std::vector<float>());
        EXPECT_EQ(attrs.fixed_size, std::vector<float>());
        EXPECT_EQ(attrs.variance, std::vector<float>());
        EXPECT_FALSE(attrs.clip);
        EXPECT_FALSE(attrs.flip);
        EXPECT_TRUE(attrs.scale_all_sizes);
        EXPECT_FLOAT_EQ(attrs.step, 0.0f);
        EXPECT_FLOAT_EQ(attrs.offset, 0.0f);
    }

    template <class T = TOp, typename std::enable_if<std::is_same<T, op::v8::PriorBox>::value>::type* = nullptr>
    static void check_def_attrs(const TOp* const op) {
        const auto& attrs = op->get_attrs();

        EXPECT_EQ(attrs.min_size, std::vector<float>());
        EXPECT_EQ(attrs.max_size, std::vector<float>());
        EXPECT_EQ(attrs.aspect_ratio, std::vector<float>());
        EXPECT_EQ(attrs.density, std::vector<float>());
        EXPECT_EQ(attrs.fixed_ratio, std::vector<float>());
        EXPECT_EQ(attrs.fixed_size, std::vector<float>());
        EXPECT_EQ(attrs.variance, std::vector<float>());
        EXPECT_FALSE(attrs.clip);
        EXPECT_FALSE(attrs.flip);
        EXPECT_TRUE(attrs.scale_all_sizes);
        EXPECT_TRUE(attrs.min_max_aspect_ratios_order);
        EXPECT_FLOAT_EQ(attrs.step, 0.0f);
        EXPECT_FLOAT_EQ(attrs.offset, 0.0f);
    }

    Attrs attrs;
};

TYPED_TEST_SUITE_P(PriorBoxTest);

TYPED_TEST_P(PriorBoxTest, default_ctor) {
    const auto output_size = std::make_shared<Parameter>(element::i32, PartialShape{2});
    const auto image_size = std::make_shared<Parameter>(element::i32, Shape{2});

    const auto op = this->make_op();
    op->set_arguments(OutputVector{output_size, image_size});
    op->set_attrs({});
    op->validate_and_infer_types();

    this->check_def_attrs(op.get());
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, -1}));
}

TYPED_TEST_P(PriorBoxTest, simple_inference) {
    const auto output_size = Constant::create(element::i8, Shape{2}, {2, 5});
    const auto image_size = Constant::create(element::i8, Shape{2}, {300, 300});

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 200}));
}

TYPED_TEST_P(PriorBoxTest, inputs_dynamic_rank) {
    const auto output_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto image_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, input_output_size_is_dynamic_rank) {
    const auto output_size = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());
    const auto image_size = Constant::create(element::u64, Shape{2}, {300, 300});

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, input_output_size_is_static_rank_with_dynamic_dims) {
    auto out_size_shape = PartialShape::dynamic(1);
    set_shape_symbols(out_size_shape);

    const auto output_size = std::make_shared<Parameter>(element::u32, out_size_shape);
    const auto image_size = Constant::create(element::u32, Shape{2}, {300, 300});

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, input_image_size_is_dynamic_rank) {
    const auto output_size = Constant::create(element::u8, Shape{2}, {32, 32});
    const auto image_size = std::make_shared<Parameter>(element::u8, PartialShape::dynamic());

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 20480}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, input_image_size_is_static_rank_dynamic_dim) {
    this->attrs.flip = true;

    auto img_size_shape = PartialShape::dynamic(1);
    set_shape_symbols(img_size_shape);

    const auto output_size = Constant::create(element::u16, Shape{2}, {32, 32});
    const auto image_size = std::make_shared<Parameter>(element::u16, img_size_shape);

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 32768}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, inputs_are_interval_shapes) {
    auto out_size_shape = PartialShape{{0, 10}};
    auto img_size_shape = PartialShape{{1, 5}};
    auto symbol = set_shape_symbols(out_size_shape);
    set_shape_symbols(img_size_shape, symbol);

    const auto output_size = std::make_shared<Parameter>(element::u64, out_size_shape);
    const auto image_size = std::make_shared<Parameter>(element::u64, img_size_shape);

    const auto op = this->make_op(output_size, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, -1}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, preseve_values_and_symbols_on_inputs) {
    auto out_size_shape = PartialShape{6, 8};
    out_size_shape[0].set_symbol(std::make_shared<Symbol>());

    const auto output_size = std::make_shared<Parameter>(element::i16, out_size_shape);
    const auto image_size = Constant::create(element::i16, Shape{2}, {300, 300});
    const auto out_shape_of = std::make_shared<ShapeOf>(output_size);

    const auto op = this->make_op(out_shape_of, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 960}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, preseve_partial_values_and_symbols_on_inputs) {
    auto out_size_shape = PartialShape{{1, 4}, {5, 10}};
    set_shape_symbols(out_size_shape);

    const auto output_size = std::make_shared<Parameter>(element::u64, out_size_shape);
    const auto image_size = Constant::create(element::u64, Shape{2}, {300, 300});
    const auto out_shape_of = std::make_shared<ShapeOf>(output_size);

    const auto op = this->make_op(out_shape_of, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, {100, 800}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, preseve_partial_values_inf_bound) {
    auto out_size_shape = PartialShape{{1, 4}, {5, -1}};
    set_shape_symbols(out_size_shape);

    const auto output_size = std::make_shared<Parameter>(element::u64, out_size_shape);
    const auto image_size = Constant::create(element::u64, Shape{2}, {300, 300});
    const auto out_shape_of = std::make_shared<ShapeOf>(output_size);

    const auto op = this->make_op(out_shape_of, image_size, this->attrs);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, {100, -1}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(PriorBoxTest, out_size_input_not_integer) {
    const auto output_size = Constant::create(element::f16, Shape{2}, {5, 5});
    const auto image_size = Constant::create(element::i16, Shape{2}, {300, 300});

    OV_EXPECT_THROW(std::ignore = this->make_op(output_size, image_size, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("output size input must be an integral number"));
}

TYPED_TEST_P(PriorBoxTest, img_size_input_not_integer) {
    const auto output_size = Constant::create(element::i16, Shape{2}, {5, 5});
    const auto image_size = Constant::create(element::bf16, Shape{2}, {300, 300});

    OV_EXPECT_THROW(std::ignore = this->make_op(output_size, image_size, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("image input must be an integral number"));
}

TYPED_TEST_P(PriorBoxTest, out_and_img_size_inputs_ranks_not_compatible) {
    const auto output_size = std::make_shared<Parameter>(element::u64, PartialShape{2});
    const auto image_size = std::make_shared<Parameter>(element::u64, PartialShape{2, 1});

    OV_EXPECT_THROW(std::ignore = this->make_op(output_size, image_size, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("output size input rank 1 must match image shape input rank 2"));
}

TYPED_TEST_P(PriorBoxTest, out_and_img_size_same_rank_not_1d) {
    const auto output_size = std::make_shared<Parameter>(element::u64, PartialShape{2, 1});
    const auto image_size = std::make_shared<Parameter>(element::u64, PartialShape{2, 1});

    OV_EXPECT_THROW(std::ignore = this->make_op(output_size, image_size, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("output size input rank 2 must match image shape input rank 2 and both must be 1-D"));
}

TYPED_TEST_P(PriorBoxTest, out_size_input_not_two_elements_tensor) {
    const auto output_size = std::make_shared<Parameter>(element::u64, PartialShape{5, 5, 5});
    const auto out_shape_of = std::make_shared<ShapeOf>(output_size);
    const auto image_size = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());

    OV_EXPECT_THROW(std::ignore = this->make_op(out_shape_of, image_size, this->attrs),
                    NodeValidationFailure,
                    HasSubstr("Output size must have two elements"));
}

REGISTER_TYPED_TEST_SUITE_P(PriorBoxTest,
                            default_ctor,
                            simple_inference,
                            inputs_dynamic_rank,
                            input_output_size_is_dynamic_rank,
                            input_output_size_is_static_rank_with_dynamic_dims,
                            input_image_size_is_dynamic_rank,
                            input_image_size_is_static_rank_dynamic_dim,
                            inputs_are_interval_shapes,
                            preseve_values_and_symbols_on_inputs,
                            preseve_partial_values_and_symbols_on_inputs,
                            preseve_partial_values_inf_bound,
                            out_size_input_not_integer,
                            img_size_input_not_integer,
                            out_and_img_size_inputs_ranks_not_compatible,
                            out_and_img_size_same_rank_not_1d,
                            out_size_input_not_two_elements_tensor);

using PriorBoxTypes = Types<op::v0::PriorBox, op::v8::PriorBox>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, PriorBoxTest, PriorBoxTypes);
