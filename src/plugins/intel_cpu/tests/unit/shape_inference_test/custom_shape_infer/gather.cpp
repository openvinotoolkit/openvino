// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using TestParams = std::tuple<int32_t, unit_test::ShapeVector, StaticShape>;

template <typename TGather>
class CpuShapeInferenceGatherTest : public unit_test::OpCpuShapeInferenceTest<TGather> {
protected:
    void SetUp() override {
        this->output_shapes.resize(0);
    }

    std::shared_ptr<TGather> make_gather(const unit_test::ShapeVector& shapes, const int32_t* const axis_val_ptr = nullptr) {
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
    std::vector<TestParams>{make_tuple(0, unit_test::ShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({2, 2, 2})),
                            make_tuple(1, unit_test::ShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({3, 2, 2})),
                            make_tuple(-1, unit_test::ShapeVector{{3, 2}, {2, 2}, {1}}, StaticShape({3, 2, 2})),
                            make_tuple(0, unit_test::ShapeVector{{3, 2, 4}, {2, 1, 2}, {1}}, StaticShape({2, 1, 2, 2, 4})),
                            make_tuple(1, unit_test::ShapeVector{{3, 2, 4}, {2, 1, 2}, {1}}, StaticShape({3, 2, 1, 2, 4})),
                            make_tuple(-1, unit_test::ShapeVector{{3, 2, 4}, {2, 1, 2}, {}}, StaticShape({3, 2, 2, 1, 2})),
                            make_tuple(-2, unit_test::ShapeVector{{3, 2, 4}, {2, 1, 2}, {}}, StaticShape({3, 2, 1, 2, 4}))};

TYPED_TEST_SUITE_P(CpuShapeInferenceGatherTest);

TYPED_TEST_P(CpuShapeInferenceGatherTest, axis_const) {
    for (auto&& params : GatherTestParams) {
        std::tie(this->axis_val, this->input_shapes, this->exp_shape) = params;

        auto op = this->make_gather(this->input_shapes, &this->axis_val);
        this->output_shapes = {this->exp_shape};
        unit_test::cpu_test_shape_infer(op.get(), this->input_shapes, this->output_shapes);
    }
}

TYPED_TEST_P(CpuShapeInferenceGatherTest, axis_in_const_map) {
    for (auto&& params : GatherTestParams) {
        std::tie(this->axis_val, this->input_shapes, this->exp_shape) = params;

        auto op = this->make_gather(this->input_shapes);
        auto axis_tensor = ov::Tensor(element::i32, ov::Shape{1}, &this->axis_val);

        this->output_shapes = {this->exp_shape};
        unit_test::cpu_test_shape_infer(op.get(), this->input_shapes, this->output_shapes, {{2, axis_tensor}});
    }
}

REGISTER_TYPED_TEST_SUITE_P(CpuShapeInferenceGatherTest, axis_const, axis_in_const_map);
using GatherTypes = Types<op::v1::Gather, op::v7::Gather, op::v8::Gather>;
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer, CpuShapeInferenceGatherTest, GatherTypes);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
