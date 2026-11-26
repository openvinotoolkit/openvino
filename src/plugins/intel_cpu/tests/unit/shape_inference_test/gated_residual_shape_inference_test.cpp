// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/parameter.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;

namespace {

using Gr = op::internal::GatedResidual;

// C = D = 4 (C*D = 16), L = 8.
constexpr size_t C = 4;
constexpr size_t D = 4;
constexpr size_t L = 8;

std::shared_ptr<Parameter> param(const element::Type& et) {
    return std::make_shared<Parameter>(et, PartialShape::dynamic());
}

std::shared_ptr<Gr> make_read(bool compressed, bool write_gate) {
    const auto w_et = compressed ? element::u4 : element::f32;
    OutputVector args{param(element::f32), param(element::f32), param(w_et), param(w_et)};
    if (compressed) {
        args.push_back(param(element::f16));
        args.push_back(param(element::f16));
    }
    if (write_gate) {
        args.push_back(param(w_et));
        if (compressed) {
            args.push_back(param(element::f16));
        }
    }
    return std::make_shared<Gr>(args, Gr::Config{Gr::Mode::READ, C, D, L, 1e-6F, write_gate});
}

std::shared_ptr<Gr> make_write() {
    return std::make_shared<Gr>(OutputVector{param(element::f32), param(element::f32), param(element::f32)},
                                Gr::Config{Gr::Mode::WRITE, C, D, 0, 1e-6F, false});
}

}  // namespace

TEST(StaticShapeInferenceTest, GatedResidualRead) {
    const auto op = make_read(false, false);
    const auto output_shapes = shape_inference(
        op.get(),
        std::vector<StaticShape>{StaticShape{2, 3, 16}, StaticShape{16}, StaticShape{8, 16}, StaticShape{16, 8}});
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 3, 4}));
}

TEST(StaticShapeInferenceTest, GatedResidualReadWithWriteGate) {
    const auto op = make_read(false, true);
    const auto output_shapes = shape_inference(op.get(),
                                               std::vector<StaticShape>{StaticShape{5, 16},
                                                                        StaticShape{16},
                                                                        StaticShape{8, 16},
                                                                        StaticShape{16, 8},
                                                                        StaticShape{4, 16}});
    ASSERT_EQ(output_shapes.size(), 2);
    EXPECT_EQ(output_shapes[0], StaticShape({5, 4}));
    EXPECT_EQ(output_shapes[1], StaticShape({5, 4}));
}

TEST(StaticShapeInferenceTest, GatedResidualReadCompressedGroups) {
    const auto op = make_read(true, false);
    const auto output_shapes = shape_inference(op.get(),
                                               std::vector<StaticShape>{StaticShape{2, 3, 16},
                                                                        StaticShape{16},
                                                                        StaticShape{8, 16},
                                                                        StaticShape{16, 8},
                                                                        StaticShape{8, 2},
                                                                        StaticShape{16, 2}});
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 3, 4}));
}

TEST(StaticShapeInferenceTest, GatedResidualReadCompressedSingleElementScales) {
    const auto op = make_read(true, false);
    const auto output_shapes = shape_inference(op.get(),
                                               std::vector<StaticShape>{StaticShape{2, 3, 16},
                                                                        StaticShape{16},
                                                                        StaticShape{8, 16},
                                                                        StaticShape{16, 8},
                                                                        StaticShape{1},
                                                                        StaticShape{1, 1}});
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 3, 4}));
}

TEST(StaticShapeInferenceTest, GatedResidualReadCompressedInvalidScaleShape) {
    const auto op = make_read(true, false);
    EXPECT_THROW(shape_inference(op.get(),
                                 std::vector<StaticShape>{StaticShape{2, 3, 16},
                                                          StaticShape{16},
                                                          StaticShape{8, 16},
                                                          StaticShape{16, 8},
                                                          StaticShape{2},
                                                          StaticShape{16, 2}}),
                 NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, GatedResidualReadCompressedGroupsMustDivideK) {
    const auto op = make_read(true, false);
    EXPECT_THROW(shape_inference(op.get(),
                                 std::vector<StaticShape>{StaticShape{2, 3, 16},
                                                          StaticShape{16},
                                                          StaticShape{8, 16},
                                                          StaticShape{16, 8},
                                                          StaticShape{8, 2},
                                                          StaticShape{16, 3}}),
                 NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, GatedResidualReadInvalidResidualWidth) {
    const auto op = make_read(false, false);
    EXPECT_THROW(
        shape_inference(
            op.get(),
            std::vector<StaticShape>{StaticShape{2, 3, 12}, StaticShape{16}, StaticShape{8, 16}, StaticShape{16, 8}}),
        NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, GatedResidualWrite) {
    const auto op = make_write();
    const auto output_shapes =
        shape_inference(op.get(),
                        std::vector<StaticShape>{StaticShape{2, 3, 16}, StaticShape{2, 3, 4}, StaticShape{2, 3, 4}});
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 3, 16}));
}

TEST(StaticShapeInferenceTest, GatedResidualWriteMismatchedRows) {
    const auto op = make_write();
    EXPECT_THROW(
        shape_inference(op.get(),
                        std::vector<StaticShape>{StaticShape{2, 3, 16}, StaticShape{2, 2, 4}, StaticShape{2, 3, 4}}),
        NodeValidationFailure);
}

// GatedResidual must not fall back to the generic shape inference, which asks for the data of every input
// (FULL_PORT_MASK): Node::updateShapes() then skips re-acquiring the output memory after
// CompiledModel::release_memory() when the input shapes are unchanged (#39339).
TEST(StaticShapeInferenceTest, GatedResidualDoesNotUseTheFallbackShapeInference) {
    for (const auto& op : {make_read(false, false), make_read(true, true), make_write()}) {
        EXPECT_EQ(make_shape_inference(op)->get_port_mask(), EMPTY_PORT_MASK);
    }
}
