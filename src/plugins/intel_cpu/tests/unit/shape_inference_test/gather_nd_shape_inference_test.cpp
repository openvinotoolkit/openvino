// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_shape_inference.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/util/common_util.hpp"
#include "utils.hpp"
#include "utils/shape_inference/shape_inference.hpp"

using namespace ov;
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
void run_gather_nd_test(const GatherNDParams& test_params) {
    auto op = make_gather_nd<TGatherND>(test_params.batch_dims);

    ShapeVector output_shapes(1);
    shape_inference(op.get(), test_params.input_shapes, output_shapes);

    EXPECT_EQ(output_shapes[0], test_params.exp_shape)
        << "Failed for input shapes: " << ov::util::vector_to_string(test_params.input_shapes)
        << " and batch_dims = " << test_params.batch_dims << std::endl;
}
}  // namespace

template <class TGatherND>
class StaticShapeInferenceGatherNDTest : public OpStaticShapeInferenceTest<TGatherND> {};

// Output shape for V5 and V8 is the same, when batch_dims attribute is less than 2
const auto GatherNDGatherNDTestParams = std::vector<GatherNDParams>{
    // Test: batch_dims = 0
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {2}}, StaticShape{11, 12}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {2, 1}}, StaticShape{2, 3, 11, 12}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {2, 2}}, StaticShape{2, 11, 12}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {2, 5, 4}}, StaticShape{2, 5}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {2, 5, 20, 3}}, StaticShape{2, 5, 20, 12}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {6, 4, 2}}, StaticShape{6, 4, 11, 12}, 0},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 4, 2}}, StaticShape{8, 4, 11, 12}, 0},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 1}}, StaticShape{8, 6, 5, 4, 3, 11, 12}, 0},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 2}}, StaticShape{8, 6, 5, 4, 11, 12}, 0},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 3}}, StaticShape{8, 6, 5, 4, 12}, 0},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {8, 6, 5, 4, 4}}, StaticShape{8, 6, 5, 4}, 0},
    GatherNDParams{ShapeVector{{7, 3, 11}, {8, 6, 5, 4, 1}}, StaticShape{8, 6, 5, 4, 3, 11}, 0},
    // Test: batch_dims = 1
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 2}}, StaticShape{8, 12}, 1},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 2, 1}}, StaticShape{8, 2, 11, 12}, 1},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 5, 2}}, StaticShape{8, 5, 12}, 1},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 5, 3}}, StaticShape{8, 5}, 1},
    GatherNDParams{ShapeVector{{8, 3, 11, 12}, {8, 7, 4, 2}}, StaticShape{8, 7, 4, 12}, 1},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 1}}, StaticShape{7, 6, 5, 4, 11, 12}, 1},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 2}}, StaticShape{7, 6, 5, 4, 12}, 1},
    GatherNDParams{ShapeVector{{7, 3, 11, 12}, {7, 6, 5, 4, 3}}, StaticShape{7, 6, 5, 4}, 1}};

TYPED_TEST_SUITE_P(StaticShapeInferenceGatherNDTest);

TYPED_TEST_P(StaticShapeInferenceGatherNDTest, gather_nd_common_batch_dims) {
    for (const auto& params : GatherNDGatherNDTestParams) {
        run_gather_nd_test<TypeParam>(params);
    }
}

TYPED_TEST_P(StaticShapeInferenceGatherNDTest, gather_nd_common_default_ctor) {
    auto op = std::make_shared<TypeParam>();
    op->set_batch_dims(1);

    ShapeVector input_shapes{{8, 3, 11, 12}, {8, 5, 2}};
    ShapeVector output_shapes(1);

    shape_infer(op.get(), input_shapes, output_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{8, 5, 12}));
}

REGISTER_TYPED_TEST_SUITE_P(StaticShapeInferenceGatherNDTest, gather_nd_common_batch_dims, gather_nd_common_default_ctor);
using GatherNDTypes = Types<op::v5::GatherND, op::v8::GatherND>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_infer, StaticShapeInferenceGatherNDTest, GatherNDTypes);

// ------------------------------ V5 ------------------------------
class StaticShapeInferenceGatherNDV5Test : public TestWithParam<GatherNDParams> {};

TEST_P(StaticShapeInferenceGatherNDV5Test, gather_nd_v5_test1) {
    run_gather_nd_test<op::v5::GatherND>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    shape_infer,
    StaticShapeInferenceGatherNDV5Test,
    ::testing::Values(GatherNDParams{ShapeVector{{6, 4, 11, 12, 13}, {6, 4, 2}}, StaticShape{24, 13}, 2},
                      GatherNDParams{ShapeVector{{6, 4, 1, 12, 13}, {6, 4, 1, 1}}, StaticShape{24, 13}, 3},
                      GatherNDParams{ShapeVector{{6, 4, 1, 12, 13}, {6, 4, 1, 2}}, StaticShape{24}, 3}),
    PrintToStringParamName());

// ------------------------------ V8 ------------------------------
class StaticShapeInferenceGatherNDV8Test : public TestWithParam<GatherNDParams> {};

TEST_P(StaticShapeInferenceGatherNDV8Test, gather_nd_v8_test1) {
    run_gather_nd_test<op::v8::GatherND>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    shape_infer,
    StaticShapeInferenceGatherNDV8Test,
    ::testing::Values(GatherNDParams{ShapeVector{{6, 4, 11, 12, 13}, {6, 4, 2}}, StaticShape{6, 4, 13}, 2},
                      GatherNDParams{ShapeVector{{6, 4, 1, 12, 13}, {6, 4, 1, 2}}, StaticShape{6, 4, 1}, 3},
                      GatherNDParams{ShapeVector{{6, 4, 1, 12, 13}, {6, 4, 1, 1}}, StaticShape{6, 4, 1, 13}, 3}),
    PrintToStringParamName());
