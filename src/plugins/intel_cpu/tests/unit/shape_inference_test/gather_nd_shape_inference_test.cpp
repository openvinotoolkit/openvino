// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/shape_inference/shape_inference.hpp"
#include "gather_nd_shape_inference.hpp"

#include "utils.hpp"

using namespace ov;
using namespace op;
using namespace ov::intel_cpu;
using namespace testing;

namespace {
struct GatherNDParams {
    ShapeVector input_shapes;
    StaticShape exp_shape;
    size_t batch_dims;
};

template <class TGatherND>
std::shared_ptr<TGatherND> make_gather_nd(size_t batch_dims) {
    auto data_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto indicies_param = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    return std::make_shared<TGatherND>(data_param, indicies_param, batch_dims);
}

template <typename TGatherND>
void run_gather_nd_test(GatherNDParams test_params) {
    auto op = make_gather_nd<TGatherND>(test_params.batch_dims);

    ShapeVector output_shapes(1);
    shape_inference(op.get(), test_params.input_shapes, output_shapes);

    EXPECT_EQ(output_shapes[0], test_params.exp_shape)
    << "Failed for input shapes: " << ov::util::vector_to_string(test_params.input_shapes)
    << " and batch_dims = " << test_params.batch_dims << std::endl;
}
} // namespace

using GatherNDTestParams = std::tuple<ShapeVector, StaticShape, size_t>;

template <class TGatherND>
class StaticShapeInferenceGatherNDTest : public OpStaticShapeInferenceTest<TGatherND>{
protected:
    void SetUp() override {
        OpStaticShapeInferenceTest<TGatherND>::output_shapes = ShapeVector(1);
    }
    size_t batch_dims = 1;
};

// Output shape for V5 and V8 is the same, when batch_dims attribute is less than 2
const auto GatherNDGatherNDTestParams =
    std::vector<GatherNDTestParams>{
        // batch_dims = 0
        make_tuple(ShapeVector{{8, 3, 11, 12}, {2}}, StaticShape({11, 12}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {2, 1}}, StaticShape({2, 3, 11, 12}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {2, 2}}, StaticShape({2, 11, 12}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {2, 5, 4}}, StaticShape({2, 5}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {2, 5, 20, 3}}, StaticShape({2, 5, 20, 12}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {6, 4, 2}}, StaticShape({6, 4, 11, 12}), 0),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 4, 2}}, StaticShape({8, 4, 11, 12}), 0),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 1}}, StaticShape({8, 6, 5, 4, 3, 11, 12}), 0),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 2}}, StaticShape({8, 6, 5, 4, 11, 12}), 0),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 3}}, StaticShape({8, 6, 5, 4, 12}), 0),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 4}}, StaticShape({8, 6, 5, 4}), 0),
        make_tuple(ShapeVector{{7, 3, 11}, {8, 6, 5, 4, 1}}, StaticShape({8, 6, 5, 4, 3, 11}), 0),
        // batch_dims = 1
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 2}}, StaticShape({8, 12}), 1),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 2, 1}}, StaticShape({8, 2, 11, 12}), 1),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 5, 2}}, StaticShape({8, 5, 12}), 1),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 5, 3}}, StaticShape({8, 5}), 1),
        make_tuple(ShapeVector{{8, 3, 11, 12}, {8, 7, 4, 2}}, StaticShape({8, 7, 4, 12}), 1),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 1}}, StaticShape({7, 6, 5, 4, 11, 12}), 1),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 2}}, StaticShape({7, 6, 5, 4, 12}), 1),
        make_tuple(ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 3}}, StaticShape({7, 6, 5, 4}), 1)
    };

TYPED_TEST_SUITE_P(StaticShapeInferenceGatherNDTest);

TYPED_TEST_P(StaticShapeInferenceGatherNDTest, gather_nd_batch_dims_1) {
    for (auto&& params : GatherNDGatherNDTestParams) {
        std::tie(this->input_shapes, this->exp_shape, this->batch_dims) = params;

        auto op = make_gather_nd<TypeParam>(this->batch_dims);
        shape_inference(op.get(), this->input_shapes, this->output_shapes);

        EXPECT_EQ(this->output_shapes.front(), this->exp_shape)
        << "Failed for input shapes: " << ov::util::vector_to_string(this->input_shapes)
        << " and batch_dims = " << this->batch_dims << std::endl;
    }
}

REGISTER_TYPED_TEST_SUITE_P(StaticShapeInferenceGatherNDTest, gather_nd_batch_dims_1);
using GatherNDTypes = Types<op::v5::GatherND, op::v8::GatherND>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer, StaticShapeInferenceGatherNDTest, GatherNDTypes);


// ------------------------------ V5 ------------------------------
class StaticShapeInferenceGatherNDV5Test : public TestWithParam<GatherNDParams> {};

TEST_P(StaticShapeInferenceGatherNDV5Test, gather_nd_v5_test1) {
    run_gather_nd_test<op::v5::GatherND>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(gather_nd_v5_tests,
                         StaticShapeInferenceGatherNDV5Test,
                         ::testing::Values(GatherNDParams{ShapeVector{{6, 4, 11, 12, 13}, {6, 4, 2}}, StaticShape{24, 13}, 2}),
                         PrintToStringParamName());


// ------------------------------ V8 ------------------------------
class StaticShapeInferenceGatherNDV8Test : public TestWithParam<GatherNDParams> {};

TEST_P(StaticShapeInferenceGatherNDV8Test, gather_nd_v8_test1) {
    run_gather_nd_test<op::v8::GatherND>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(gather_nd_v8_tests,
                         StaticShapeInferenceGatherNDV8Test,
                         ::testing::Values(GatherNDParams{ShapeVector{{6, 4, 11, 12, 13}, {6, 4, 2}}, StaticShape{6, 4, 13}, 2}),
                         PrintToStringParamName());
