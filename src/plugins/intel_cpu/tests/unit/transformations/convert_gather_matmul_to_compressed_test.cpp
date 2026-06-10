// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

using namespace testing;
using namespace ov::pass;

namespace {
struct ConvertBGMToCompressedParams {
    ov::element::Type compressed_type;
    bool grouped;
    ov::PartialShape in_shape;
    ov::Shape wei_shape;
    ov::Shape scale_zp_shape;
    ov::Shape index_shape;
};

class ConvertBGMToCompressed : public testing::WithParamInterface<ConvertBGMToCompressedParams>,
                               public TransformationTestsF {};

TEST_P(ConvertBGMToCompressed, ConvertBGMToCompressedTest) {
    const auto& params = GetParam();
    ov::element::Type compressed_type_ = params.compressed_type;
    bool grouped_ = params.grouped;
    ov::PartialShape in_shape_ = params.in_shape;
    ov::Shape wei_shape_ = params.wei_shape;
    ov::Shape scale_zp_shape_ = params.scale_zp_shape;
    ov::Shape index_shape_ = params.index_shape;
    const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
    const std::vector<ov::element::Type> supported_weights_types{compressed_type_};

    manager.register_pass<ConvertGatherMatmulToGatherMatmulCompressed>(supported_activation_types,
                                                                       supported_weights_types);
    auto weight_reshaped_dims = [&]() {
        std::vector<int64_t> wei_reshaped;
        for (size_t i = 0; i < wei_shape_.size() - 2; ++i) {
            wei_reshaped.push_back(static_cast<int64_t>(wei_shape_[i]));
        }
        int64_t combined_dim = static_cast<int64_t>(wei_shape_[wei_shape_.size() - 2]) *
                               static_cast<int64_t>(wei_shape_[wei_shape_.size() - 1]);
        wei_reshaped.push_back(combined_dim);
        return wei_reshaped;
    };

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape_);
        auto weights_const = ov::op::v0::Constant::create(compressed_type_, wei_shape_, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);

        auto zp_const = ov::op::v0::Constant::create(compressed_type_, scale_zp_shape_, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);

        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape_, {1});
        std::shared_ptr<ov::op::Op> wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);

        if (grouped_) {
            std::vector<int64_t> wei_reshaped = weight_reshaped_dims();
            auto reshape_pattern =
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{wei_reshaped.size()}, wei_reshaped);
            wei_scale = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_pattern, false);
        }

        auto index = ov::op::v0::Constant::create(ov::element::i32, index_shape_, {1});

        auto bgm = std::make_shared<ov::op::internal::GatherMatmul>(input, wei_scale, index);
        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape_);
        auto reshape_dims = wei_shape_;
        if (grouped_) {
            std::vector<int64_t> wei_reshaped = weight_reshaped_dims();
            reshape_dims = ov::Shape(wei_reshaped.begin(), wei_reshaped.end());
        }
        auto weights_const = ov::op::v0::Constant::create(compressed_type_, reshape_dims, {1});

        auto scale_zp_shape = scale_zp_shape_;
        if (grouped_) {
            scale_zp_shape.pop_back();
        }
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape, {1});
        auto zp_const = ov::op::v0::Constant::create(compressed_type_, scale_zp_shape, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, index_shape_, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed = std::make_shared<ov::op::internal::GatherMatmulCompressed>(input,
                                                                                         weights_const,
                                                                                         index,
                                                                                         bias,
                                                                                         scale_const,
                                                                                         zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

const auto params = std::vector<ConvertBGMToCompressedParams>{
    {ov::element::u8, false, ov::PartialShape{8, 10, 2048}, ov::Shape{128, 5, 2048}, ov::Shape{128, 5, 1}, {10, 8}},
    // grouped
    {ov::element::u8,
     true,
     ov::PartialShape{8, 10, 2048},
     ov::Shape{128, 5, 16, 128},
     ov::Shape{128, 5, 16, 1},
     {10, 8}},
    // grouped with output channel 1
    {ov::element::u8,
     true,
     ov::PartialShape{8, 10, 2048},
     ov::Shape{128, 1, 16, 128},
     ov::Shape{128, 1, 16, 1},
     {10, 8}},

    {ov::element::u4, false, ov::PartialShape{-1, 10, 512}, ov::Shape{32, 5, 512}, ov::Shape{32, 5, 1}, {10, 8}},
    // grouped
    {ov::element::u4, true, ov::PartialShape{-1, 10, 512}, ov::Shape{32, 5, 4, 128}, ov::Shape{32, 5, 4, 1}, {10, 8}},
    // grouped with output channel 1
    {ov::element::u4, true, ov::PartialShape{-1, 10, 512}, ov::Shape{32, 1, 4, 128}, ov::Shape{32, 1, 4, 1}, {10, 8}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests, ConvertBGMToCompressed, ::testing::ValuesIn(params));
