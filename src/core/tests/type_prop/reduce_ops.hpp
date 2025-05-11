// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

struct ReduceParams {
    PartialShape data_ps;
    element::Type data_et;
    Shape axes_ps;
    std::vector<int64_t> axes;
    element::Type axes_et;
    bool keep_dims;
};

template <class T>
static std::shared_ptr<Node> makeReduceOp(const ReduceParams& p, bool axes_as_param = false) {
    auto in_data = make_shared<ov::op::v0::Parameter>(p.data_et, p.data_ps);
    shared_ptr<Node> in_axes;
    if (axes_as_param) {
        in_axes = make_shared<ov::op::v0::Parameter>(p.axes_et, p.axes_ps);
    } else {
        if (shape_size(p.axes_ps) != p.axes.size()) {
            OPENVINO_THROW("Axes shape does not match with axes elements");
        }
        in_axes = make_shared<ov::op::v0::Constant>(p.axes_et, p.axes_ps, p.axes);
    }
    return make_shared<T>(in_data, in_axes, p.keep_dims);
}

template <class T>
class ReduceTest : public testing::Test {};

TYPED_TEST_SUITE_P(ReduceTest);

TYPED_TEST_P(ReduceTest, reduce_default_ctor) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = true;

    const auto data = make_shared<ov::op::v0::Parameter>(data_et, data_ps);
    const auto in_axes = make_shared<ov::op::v0::Parameter>(axes_et, axes_ps);

    auto op = std::make_shared<TypeParam>();
    op->set_arguments(OutputVector{data, in_axes});
    op->set_keep_dims(keep_dims);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);

    EXPECT_EQ(op->get_keep_dims(), keep_dims);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{3};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_keep_dims) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = true;

    PartialShape out_ps{3, 1, 1};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_duplicated_axes) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 1};

    bool keep_dims = false;

    PartialShape out_ps{3, 5};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    EXPECT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_keep_dims_duplicated_axes) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 1};

    bool keep_dims = true;

    PartialShape out_ps{3, 1, 5};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    EXPECT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_scalar_axis) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1};

    bool keep_dims = false;

    PartialShape out_ps{3, 5};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_basic_shape_infer_axes_as_param) {
    PartialShape data_ps{3, 4, 5};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i32;
    std::vector<int64_t> axes;

    bool keep_dims = false;

    PartialShape out_ps{PartialShape::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    bool axes_as_param = true;
    auto reduce_op = makeReduceOp<TypeParam>(params, axes_as_param);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_static) {
    PartialShape data_ps{3, 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{3, Dimension::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_static_keep_dims) {
    PartialShape data_ps{3, 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = true;

    PartialShape out_ps{3, 1, 1, Dimension::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_not_static) {
    PartialShape data_ps{Dimension::dynamic(), 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = false;

    PartialShape out_ps{Dimension::dynamic(), 4};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_reduced_axes_not_static_keep_dims) {
    PartialShape data_ps{Dimension::dynamic(), 4, 5, Dimension::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = true;

    PartialShape out_ps{Dimension::dynamic(), 4, 1, 1};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, reduce_dynamic_shape_data) {
    PartialShape data_ps{PartialShape::dynamic()};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 2};

    bool keep_dims = false;

    PartialShape out_ps{PartialShape::dynamic()};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    ASSERT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
}

TYPED_TEST_P(ReduceTest, dynamic_interval_symboled_shape_data_axes_const) {
    using namespace testing;

    PartialShape data_ps{-1, -1, 1, 1, 6, 16, {-1, 8}, {-1, 18}, {4, -1}, {14, -1}, {3, 9}, {13, 19}};
    element::Type data_et = element::dynamic;

    auto symbols = set_shape_symbols(data_ps);

    Shape axes_ps{6};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 3, 5, 7, 9, 11};

    bool keep_dims = false;

    PartialShape out_ps{-1, 1, 6, {-1, 8}, {4, -1}, {3, 9}};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    EXPECT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
    EXPECT_THAT(get_shape_symbols(reduce_op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[2], symbols[4], symbols[6], symbols[8], symbols[10]));
}

TYPED_TEST_P(ReduceTest, dynamic_interval_symboled_shape_data_axes_const_keep_dims) {
    using namespace testing;

    PartialShape data_ps{-1, -1, 1, 1, 6, 16, {-1, 8}, {-1, 18}, {4, -1}, {14, -1}, {3, 9}, {13, 19}};
    element::Type data_et = element::dynamic;

    auto symbols = set_shape_symbols(data_ps);

    Shape axes_ps{6};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{1, 3, 5, 7, 9, 11};

    bool keep_dims = true;

    PartialShape out_ps{-1, 1, 1, 1, 6, 1, {-1, 8}, 1, {4, -1}, 1, {3, 9}, 1};

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    auto reduce_op = makeReduceOp<TypeParam>(params);
    EXPECT_EQ(reduce_op->get_output_partial_shape(0), out_ps);
    EXPECT_THAT(get_shape_symbols(reduce_op->get_output_partial_shape(0)),
                ElementsAre(symbols[0],
                            nullptr,
                            symbols[2],
                            nullptr,
                            symbols[4],
                            nullptr,
                            symbols[6],
                            nullptr,
                            symbols[8],
                            nullptr,
                            symbols[10],
                            nullptr));
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axis_out_of_range) {
    PartialShape data_ps{1, 2, 3};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{2, 3};

    bool keep_dims = false;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid axes values not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "out of the tensor rank range");
    } catch (...) {
        FAIL() << "Axes input values validation check failed for unexpected reason";
    }
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axes_shape) {
    PartialShape data_ps{1, 2, 3};
    element::Type data_et = element::dynamic;

    Shape axes_ps{2, 1};
    element::Type axes_et = element::i64;
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid shape of axes input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Axes input must be a scalar or 1D input.");
    } catch (...) {
        FAIL() << "Axes input shape validation check failed for unexpected reason";
    }
}

TYPED_TEST_P(ReduceTest, reduce_invalid_axes_et) {
    element::Type data_et = element::dynamic;
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::f32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid element type of axes input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of axes input must be integer.");
    } catch (...) {
        FAIL() << "Axes input element type validation check failed for unexpected reason";
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceTest,
                            reduce_default_ctor,
                            reduce_basic_shape_infer,
                            reduce_basic_shape_infer_keep_dims,
                            reduce_basic_shape_infer_duplicated_axes,
                            reduce_basic_shape_infer_keep_dims_duplicated_axes,
                            reduce_basic_shape_infer_scalar_axis,
                            reduce_basic_shape_infer_axes_as_param,
                            reduce_dynamic_shape_data,
                            reduce_dynamic_shape_reduced_axes_static,
                            reduce_dynamic_shape_reduced_axes_static_keep_dims,
                            reduce_dynamic_shape_reduced_axes_not_static,
                            reduce_dynamic_shape_reduced_axes_not_static_keep_dims,
                            dynamic_interval_symboled_shape_data_axes_const_keep_dims,
                            dynamic_interval_symboled_shape_data_axes_const,
                            reduce_invalid_axis_out_of_range,
                            reduce_invalid_axes_shape,
                            reduce_invalid_axes_et);

template <class T>
class ReduceArithmeticTest : public testing::Test {};

TYPED_TEST_SUITE_P(ReduceArithmeticTest);

TYPED_TEST_P(ReduceArithmeticTest, reduce_arithmetic_invalid_data_et) {
    element::Type data_et = element::boolean;
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::i32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
    try {
        auto reduce_op = makeReduceOp<TypeParam>(params);
        FAIL() << "Invalid element type of data input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of data input must be numeric.");
    } catch (...) {
        FAIL() << "Data input element type validation check failed for unexpected reason";
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceArithmeticTest, reduce_arithmetic_invalid_data_et);

template <class T>
class ReduceLogicalTest : public testing::Test {};

TYPED_TEST_SUITE_P(ReduceLogicalTest);

TYPED_TEST_P(ReduceLogicalTest, reduce_logical_invalid_data_et) {
    std::vector<element::Type> element_types{element::f32, element::i32, element::u32};
    PartialShape data_ps{1, 2, 3};

    element::Type axes_et = element::i32;
    Shape axes_ps{2};
    std::vector<int64_t> axes{0, 1};

    bool keep_dims = true;

    for (const auto& data_et : element_types) {
        const ReduceParams params{data_ps, data_et, axes_ps, axes, axes_et, keep_dims};
        try {
            auto reduce_op = makeReduceOp<TypeParam>(params);
            FAIL() << "Invalid element type of data input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "Element type of data input must be boolean.");
        } catch (...) {
            FAIL() << "Data input element type validation check failed for unexpected reason";
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceLogicalTest, reduce_logical_invalid_data_et);
