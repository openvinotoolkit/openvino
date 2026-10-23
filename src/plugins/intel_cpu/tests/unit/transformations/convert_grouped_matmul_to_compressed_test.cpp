// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_compressed.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

using namespace testing;
using namespace ov::pass;

namespace {

struct ConvertGMMToCompressedParams {
    ov::element::Type compressed_type;
    bool grouped;
    bool with_offsets;          // 2D x 3D form when true, 3D x 3D otherwise
    ov::PartialShape in_shape;  // mat_a
    ov::Shape wei_shape;        // [G, N, K] or [G, N, K/gs, gs] when grouped
    ov::Shape scale_zp_shape;
};

// Builds the dequantization subgraph feeding mat_b and wraps it into a v17::GroupedMatMul.
class ConvertGMMToCompressed : public testing::WithParamInterface<ConvertGMMToCompressedParams>,
                               public TransformationTestsF {};

TEST_P(ConvertGMMToCompressed, ConvertGMMToCompressedTest) {
    const auto& params = GetParam();
    const auto compressed_type = params.compressed_type;
    const bool grouped = params.grouped;
    const bool with_offsets = params.with_offsets;
    const auto in_shape = params.in_shape;
    const auto wei_shape = params.wei_shape;
    const auto scale_zp_shape = params.scale_zp_shape;
    const std::vector<ov::element::Type> supported_weights_types{compressed_type};

    manager.register_pass<ConvertGroupedMatMulToGroupedMatMulCompressed>(supported_weights_types);

    // [G, N, K/gs, gs] is flattened back to [G, N, K] before feeding GroupedMatMul, which only
    // accepts a rank-3 mat_b.
    auto weight_reshaped_dims = [&]() {
        std::vector<int64_t> wei_reshaped;
        for (size_t i = 0; i < wei_shape.size() - 2; ++i) {
            wei_reshaped.push_back(static_cast<int64_t>(wei_shape[i]));
        }
        wei_reshaped.push_back(static_cast<int64_t>(wei_shape[wei_shape.size() - 2]) *
                               static_cast<int64_t>(wei_shape[wei_shape.size() - 1]));
        return wei_reshaped;
    };

    const size_t G = wei_shape[0];

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
        auto weights_const = ov::op::v0::Constant::create(compressed_type, wei_shape, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);

        auto zp_const = ov::op::v0::Constant::create(compressed_type, scale_zp_shape, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);

        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape, {1});
        std::shared_ptr<ov::op::Op> wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);

        if (grouped) {
            auto wei_reshaped = weight_reshaped_dims();
            auto reshape_pattern =
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{wei_reshaped.size()}, wei_reshaped);
            wei_scale = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_pattern, false);
        }

        std::shared_ptr<ov::Node> gmm;
        ov::ParameterVector model_params{input};
        if (with_offsets) {
            auto offsets = std::make_shared<ov::op::v0::Parameter>(
                ov::element::i32,
                ov::PartialShape{ov::Dimension(static_cast<ov::Dimension::value_type>(G))});
            model_params.push_back(offsets);
            gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, wei_scale, offsets);
        } else {
            gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, wei_scale);
        }
        model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, model_params);
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
        auto reshape_dims = wei_shape;
        if (grouped) {
            auto wei_reshaped = weight_reshaped_dims();
            reshape_dims = ov::Shape(wei_reshaped.begin(), wei_reshaped.end());
        }
        auto weights_const = ov::op::v0::Constant::create(compressed_type, reshape_dims, {1});

        auto ref_scale_zp_shape = scale_zp_shape;
        if (grouped) {
            ref_scale_zp_shape.pop_back();
        }
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ref_scale_zp_shape, {1});
        auto zp_const = ov::op::v0::Constant::create(compressed_type, ref_scale_zp_shape, {1});

        std::shared_ptr<ov::Node> gmm_compressed;
        ov::ParameterVector model_params{input};
        if (with_offsets) {
            auto offsets = std::make_shared<ov::op::v0::Parameter>(
                ov::element::i32,
                ov::PartialShape{ov::Dimension(static_cast<ov::Dimension::value_type>(G))});
            model_params.push_back(offsets);
            gmm_compressed = std::make_shared<ov::op::internal::GroupedMatMulCompressed>(input,
                                                                                         weights_const,
                                                                                         offsets,
                                                                                         scale_const,
                                                                                         zp_const);
        } else {
            gmm_compressed =
                ov::op::internal::GroupedMatMulCompressed::make_3d(input, weights_const, scale_const, zp_const);
        }

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gmm_compressed}, model_params);
    }
}

const auto params = std::vector<ConvertGMMToCompressedParams>{
    // 3D x 3D
    {ov::element::u8, false, false, ov::PartialShape{4, 10, 2048}, ov::Shape{4, 128, 2048}, ov::Shape{4, 128, 1}},
    {ov::element::u8, true, false, ov::PartialShape{4, 10, 2048}, ov::Shape{4, 128, 16, 128}, ov::Shape{4, 128, 16, 1}},
    {ov::element::u4, false, false, ov::PartialShape{4, -1, 512}, ov::Shape{4, 32, 512}, ov::Shape{4, 32, 1}},
    // 2D x 3D
    {ov::element::u8, false, true, ov::PartialShape{-1, 2048}, ov::Shape{4, 128, 2048}, ov::Shape{4, 128, 1}},
    {ov::element::u8, true, true, ov::PartialShape{-1, 2048}, ov::Shape{4, 128, 16, 128}, ov::Shape{4, 128, 16, 1}},
    {ov::element::u4, true, true, ov::PartialShape{-1, 512}, ov::Shape{4, 32, 4, 128}, ov::Shape{4, 32, 4, 1}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests, ConvertGMMToCompressed, ::testing::ValuesIn(params));

// A supports-predicate returning false must leave the original GroupedMatMul untouched.
TEST_F(TransformationTestsF, ConvertGMMToCompressedVetoedByPredicate) {
    const auto compressed_type = ov::element::u8;
    const ov::Shape wei_shape{4, 128, 2048};
    const ov::Shape scale_zp_shape{4, 128, 1};
    const ov::PartialShape in_shape{4, 10, 2048};

    auto build = [&]() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
        auto weights_const = ov::op::v0::Constant::create(compressed_type, wei_shape, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, wei_scale);
        return std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{input});
    };

    manager.register_pass<ConvertGroupedMatMulToGroupedMatMulCompressed>(
        std::vector<ov::element::Type>{compressed_type},
        [](const std::shared_ptr<ov::op::internal::GroupedMatMulCompressed>&, size_t, size_t, size_t) {
            return false;
        });

    model = build();
    model_ref = build();
}
