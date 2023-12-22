// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat_shape_inference.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "utils.hpp"
#include "shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using ShapeVector = std::vector<StaticShape>;
using TestParams = std::tuple<int64_t,      // concatenation axis
                              ShapeVector,  // Input shapes
                              StaticShape   // Expected shape
                              >;

class ConcatStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::Concat>,
                                       public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        std::tie(concat_axis, input_shapes, exp_shape) = GetParam();

        for (const auto& in : input_shapes) {
            params.make<op::v0::Parameter>(element::f32, in.get_shape());
        }
        op = std::make_shared<op::v0::Concat>(params.get(), concat_axis);

        output_shapes = ShapeVector(1);
    }

    int64_t concat_axis;

    pass::NodeRegistry params{};
};

/** \brief Concatenate simple 1d shapes. */
INSTANTIATE_TEST_SUITE_P(concat_1d_shapes,
                         ConcatStaticShapeInferenceTest,
                         Values(make_tuple(0, ShapeVector{{0}}, StaticShape({0})),
                                make_tuple(0, ShapeVector{{3}}, StaticShape({3})),
                                make_tuple(0, ShapeVector{{1}, {1}}, StaticShape({2})),
                                make_tuple(0, ShapeVector{{1}, {3}}, StaticShape({4})),
                                make_tuple(0, ShapeVector{{4}, {1}}, StaticShape({5})),
                                make_tuple(0, ShapeVector{{4}, {0}}, StaticShape({4})),
                                make_tuple(-1, ShapeVector{{4}, {0}, {2}}, StaticShape({6})),
                                make_tuple(-1, ShapeVector{{2}, {7}, {3}}, StaticShape({12}))),
                         PrintToStringParamName());

/** \brief Concatenate complex shapes. */
INSTANTIATE_TEST_SUITE_P(
    concat_complex_shapes,
    ConcatStaticShapeInferenceTest,
    Values(make_tuple(1, ShapeVector{{0, 0}}, StaticShape({0, 0})),
           make_tuple(1, ShapeVector{{3, 1}, {3, 2}}, StaticShape({3, 3})),
           make_tuple(0, ShapeVector{{3, 1, 2}, {3, 1, 2}}, StaticShape({6, 1, 2})),
           make_tuple(-3, ShapeVector{{3, 1, 2}, {3, 1, 2}}, StaticShape({6, 1, 2})),
           make_tuple(2, ShapeVector{{3, 1, 2}, {3, 1, 2}}, StaticShape({3, 1, 4})),
           make_tuple(-2, ShapeVector{{3, 1, 2}, {3, 1, 2}}, StaticShape({3, 2, 2})),
           make_tuple(-1, ShapeVector{{2, 5, 1, 1}, {2, 5, 1, 2}, {2, 5, 1, 2}}, StaticShape({2, 5, 1, 5})),
           make_tuple(2, ShapeVector{{2, 5, 6, 2}, {2, 5, 7, 2}, {2, 5, 1, 2}}, StaticShape({2, 5, 14, 2}))),
    PrintToStringParamName());

/** \brief Check shape_infer for concat op on static shapes. */
TEST_P(ConcatStaticShapeInferenceTest, concat_static) {
    output_shapes = shape_inference(op.get(), input_shapes);

    ASSERT_EQ(output_shapes.front(), exp_shape);
}
