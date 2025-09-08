// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using TestParams = std::tuple<int32_t, StaticShapeVector, StaticShape>;

template <class TGather>
class StaticShapeInferenceGatherTest : public OpStaticShapeInferenceTest<TGather> {
protected:
    void SetUp() override {
        OpStaticShapeInferenceTest<TGather>::output_shapes = StaticShapeVector(1);
    }

    std::shared_ptr<TGather> make_gather(const StaticShapeVector& shapes, const int32_t* const axis_val_ptr = nullptr) {
        const auto p_dims = std::vector<Dimension>(shapes[0].size(), -1);
        const auto i_dims = std::vector<Dimension>(shapes[1].size(), -1);
        auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{p_dims});
        auto indicies = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{i_dims});

        if (axis_val_ptr) {
            auto axis = op::v0::Constant::create(element::i32, ov::Shape{}, {*axis_val_ptr});
            return this->make_op(param, indicies, axis);
        } else {
            auto axis = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
            return this->make_op(param, indicies, axis);
        }
    }

    int32_t axis_val;
};

// Parameters for typed test used test case internal loop.
const auto GatherTestParams =
    std::vector<TestParams>{make_tuple(0, StaticShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({2, 2, 2})),
                            make_tuple(1, StaticShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({3, 2, 2})),
                            make_tuple(-1, StaticShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({3, 2, 2})),
                            make_tuple(0, StaticShapeVector{{3, 2, 4}, {2, 1, 2}, {1}}, StaticShape({2, 1, 2, 2, 4})),
                            make_tuple(1, StaticShapeVector{{3, 2, 4}, {2, 1, 2}, {1}}, StaticShape({3, 2, 1, 2, 4})),
                            make_tuple(-1, StaticShapeVector{{3, 2, 4}, {2, 1, 2}, {}}, StaticShape({3, 2, 2, 1, 2})),
                            make_tuple(-2, StaticShapeVector{{3, 2, 4}, {2, 1, 2}, {}}, StaticShape({3, 2, 1, 2, 4}))};

TYPED_TEST_SUITE_P(StaticShapeInferenceGatherTest);

TYPED_TEST_P(StaticShapeInferenceGatherTest, axis_const) {
    for (auto&& params : GatherTestParams) {
        std::tie(this->axis_val, this->input_shapes, this->exp_shape) = params;

        auto op = this->make_gather(this->input_shapes, &this->axis_val);

        this->output_shapes = shape_inference(op.get(), this->input_shapes);

        ASSERT_EQ(this->output_shapes.front(), this->exp_shape)
            << "Failed for axis: " << this->axis_val
            << ", input shapes: " << util::vector_to_string(this->input_shapes);
    }
}

TYPED_TEST_P(StaticShapeInferenceGatherTest, axis_in_const_map) {
    for (auto&& params : GatherTestParams) {
        std::tie(this->axis_val, this->input_shapes, this->exp_shape) = params;

        auto op = this->make_gather(this->input_shapes);
        auto axis_tensor = ov::Tensor(element::i32, ov::Shape{1}, &this->axis_val);

        this->output_shapes = shape_inference(op.get(), this->input_shapes, {{2, axis_tensor}});

        ASSERT_EQ(this->output_shapes.front(), this->exp_shape)
            << "Failed for axis: " << this->axis_val
            << ", input shapes: " << util::vector_to_string(this->input_shapes);
    }
}

REGISTER_TYPED_TEST_SUITE_P(StaticShapeInferenceGatherTest, axis_const, axis_in_const_map);
using GatherTypes = Types<op::v1::Gather, op::v7::Gather, op::v8::Gather>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer, StaticShapeInferenceGatherTest, GatherTypes);
