// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_hadamard_fc.hpp"

#include <bitset>
#include <tuple>

#include "gtest/gtest.h"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

namespace {

using FC = ov::intel_gpu::op::FullyConnectedCompressed;
using Constant = ov::op::v0::Constant;

void check_matrix_fusion(const ov::element::Type& weights_type,
                         size_t rows,
                         const ov::Shape& scale_shape,
                         const ov::Shape& zero_point_shape,
                         float zero_point,
                         bool expected) {
    const auto source = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, -1, 1024});
    const auto flat = std::make_shared<ov::op::v1::Reshape>(source, Constant::create(ov::element::i64, {2}, {-1, 1024}), false);
    std::vector<float> values(1024 * 1024);
    for (size_t row = 0; row < 1024; ++row)
        for (size_t column = 0; column < 1024; ++column)
            values[row * 1024 + column] = (std::bitset<32>(row & column).count() & 1) ? -1.f / 32.f : 1.f / 32.f;
    const auto matrix = std::make_shared<ov::op::v0::MatMul>(flat, Constant::create(ov::element::f16, {1024, 1024}, values));
    const auto restored = std::make_shared<ov::op::v1::Reshape>(matrix, std::make_shared<ov::op::v3::ShapeOf>(source), false);
    const auto fc = std::make_shared<FC>(restored,
                                         Constant::create(weights_type, {rows, 1024}, {1}),
                                         std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                         Constant::create(ov::element::f16, scale_shape, {1}),
                                         Constant::create(ov::element::f16, zero_point_shape, {zero_point}));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{source});
    ov::intel_gpu::FuseHadamardIntoFC pass;
    EXPECT_EQ(pass.run_on_model(model), expected);
    EXPECT_EQ(fc->input_value(0), expected ? source->output(0) : restored->output(0));
    EXPECT_EQ(fc->get_rt_info().count(ov::intel_gpu::int2_hadamard_block_key), expected ? 1 : 0);
}

}  // namespace

TEST(FuseHadamardFCEligibility, KeepsExplicitMatrixForUnsupportedCompressedTypes) {
    for (const auto type : {ov::element::u2, ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8}) {
        SCOPED_TRACE(type);
        check_matrix_fusion(type, 16, {16, 8}, {}, 1.f, type == ov::element::u2);
    }
}

TEST(FuseHadamardFCEligibility, KeepsExplicitMatrixForUnsupportedTernaryLayouts) {
    check_matrix_fusion(ov::element::u2, 17, {17, 8}, {}, 1.f, false);
    check_matrix_fusion(ov::element::u2, 16, {16, 1}, {}, 1.f, false);
    check_matrix_fusion(ov::element::u2, 16, {16, 8}, {16, 8}, 1.f, false);
    check_matrix_fusion(ov::element::u2, 16, {16, 8}, {}, 0.f, false);
}
