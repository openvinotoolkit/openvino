// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_compressed.hpp"

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace {
struct ConvertFCToCompressedParams {
    ov::element::Type compressed_type;
    bool with_zp;
    bool grouped;
    ov::PartialShape in_shape;
    ov::Shape wei_shape;
    ov::Shape scale_zp_shape;
};

class ConvertFCToCompressed : public testing::WithParamInterface<ConvertFCToCompressedParams>,
                              public TransformationTestsF {};

TEST_P(ConvertFCToCompressed, ConvertFCToCompressedTest) {
    const auto& params = GetParam();
    ov::element::Type compressed_type_ = params.compressed_type;
    bool with_zp_ = params.with_zp;
    bool grouped_ = params.grouped;
    ov::PartialShape in_shape_ = params.in_shape;
    ov::Shape wei_shape_ = params.wei_shape;
    ov::Shape scale_zp_shape_ = params.scale_zp_shape;
    const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
    const std::vector<ov::element::Type> supported_weights_types{compressed_type_};

    manager.register_pass<ov::pass::ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
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
        std::shared_ptr<ov::op::Op> wei_convert =
            std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);

        if (with_zp_) {
            auto zp_const = ov::op::v0::Constant::create(compressed_type_, scale_zp_shape_, {1});
            auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
            wei_convert = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        }

        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape_, {1});
        std::shared_ptr<ov::op::Op> wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);

        if (grouped_) {
            std::vector<int64_t> wei_reshaped = weight_reshaped_dims();
            auto reshape_pattern =
                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{wei_reshaped.size()}, wei_reshaped);
            wei_scale = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_pattern, false);
        }
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape_);
        auto reshape_dims = wei_shape_;
        if (grouped_) {
            std::vector<int64_t> reshaped_vec = weight_reshaped_dims();
            reshape_dims = ov::Shape(reshaped_vec.begin(), reshaped_vec.end());
        }
        auto weights_const = ov::op::v0::Constant::create(compressed_type_, reshape_dims, {1});
        auto scale_zp_shape = scale_zp_shape_;
        if (grouped_) {
            scale_zp_shape.pop_back();
        }
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);
        if (with_zp_) {
            auto zp_const = ov::op::v0::Constant::create(compressed_type_, scale_zp_shape, {1});
            fc_compressed = std::make_shared<ov::op::internal::FullyConnectedCompressed>(input,
                                                                                         weights_const,
                                                                                         bias,
                                                                                         scale_const,
                                                                                         zp_const);
        }
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

const auto params = std::vector<ConvertFCToCompressedParams>{
    {ov::element::u8, false, false, ov::PartialShape{10, 2048}, ov::Shape{5, 2048}, ov::Shape{5, 1}},
    {ov::element::u8, true, false, ov::PartialShape{10, 2048}, ov::Shape{5, 2048}, ov::Shape{5, 1}},
    // grouped
    {ov::element::u8, false, true, ov::PartialShape{10, 2048}, ov::Shape{5, 16, 128}, ov::Shape{5, 16, 1}},
    {ov::element::u8, true, true, ov::PartialShape{10, 2048}, ov::Shape{5, 16, 128}, ov::Shape{5, 16, 1}},
    // grouped with output channel 1
    {ov::element::u8, false, true, ov::PartialShape{10, 2048}, ov::Shape{1, 16, 128}, ov::Shape{1, 16, 1}},
    {ov::element::u8, true, true, ov::PartialShape{10, 2048}, ov::Shape{1, 16, 128}, ov::Shape{1, 16, 1}},

    {ov::element::u4, false, false, ov::PartialShape{-1, 512}, ov::Shape{3, 512}, ov::Shape{3, 1}},
    {ov::element::u4, true, false, ov::PartialShape{-1, 512}, ov::Shape{3, 512}, ov::Shape{3, 1}},
    // grouped
    {ov::element::u4, false, true, ov::PartialShape{-1, 512}, ov::Shape{3, 4, 128}, ov::Shape{3, 4, 1}},
    {ov::element::u4, true, true, ov::PartialShape{-1, 512}, ov::Shape{3, 4, 128}, ov::Shape{3, 4, 1}},
    // grouped with output channel 1
    {ov::element::u4, false, true, ov::PartialShape{-1, 512}, ov::Shape{1, 4, 128}, ov::Shape{1, 4, 1}},
    {ov::element::u4, true, true, ov::PartialShape{-1, 512}, ov::Shape{1, 4, 128}, ov::Shape{1, 4, 1}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests, ConvertFCToCompressed, ::testing::ValuesIn(params));

// Regression test: when the matched Transpose acts on the weights tensor and
// scale / zero-point are rank-1 per-output-channel Constants, the old code
// path tried to apply the rank-2 weight perm to a rank-1 input, which
// crashes Transpose shape inference.  The fixed `apply_transpose` lambda
// promotes rank-1 per-channel Constants to rank-2 [N, 1] without inserting a
// runtime op, while still transposing the rank-2 weights.
TEST_F(TransformationTestsF, ConvertFCToCompressedRank1ScaleZpWithTranspose) {
    const size_t OC = 5;
    const size_t IC = 2048;
    const auto compressed_type = ov::element::u8;
    const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
    const std::vector<ov::element::Type> supported_weights_types{compressed_type};

    manager.register_pass<ov::pass::ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, IC});
        // Weights laid out as [IC, OC] so that a Transpose([1, 0]) yields [OC, IC]
        // (matching the FullyConnected input layout convention).
        auto weights_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{IC, OC}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        // Rank-1 per-output-channel zero-point.
        auto zp_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{OC}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_sub = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        // Rank-1 per-output-channel scale.
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{OC}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_sub, scale_const);
        auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
        auto wei_t = std::make_shared<ov::op::v1::Transpose>(wei_scale, perm);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_t, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, IC});
        // Weights Constant stays in its original [IC, OC] layout; the transformation
        // re-clones the rank-2 Transpose with the same perm.
        auto weights_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{IC, OC}, {1});
        auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
        auto weights_t = std::make_shared<ov::op::v1::Transpose>(weights_const, perm);
        // Rank-1 [OC] -> rank-2 [OC, 1] Constant reshape baked in (no runtime op).
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{OC, 1}, {1});
        auto zp_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{OC, 1}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_t, bias, scale_const, zp_const);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// Regression test: rank-0 (scalar / per-tensor) scale and zero-point with weights
// Transpose. The old code path applied the rank-2 weight perm to a rank-0 input
// which crashes Transpose shape inference. The fixed `apply_transpose` returns
// rank-0 Constants unchanged while still transposing the rank-2 weights.
TEST_F(TransformationTestsF, ConvertFCToCompressedRank0ScaleZpWithTranspose) {
    const size_t OC = 5;
    const size_t IC = 2048;
    const auto compressed_type = ov::element::u8;
    const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
    const std::vector<ov::element::Type> supported_weights_types{compressed_type};

    manager.register_pass<ov::pass::ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, IC});
        auto weights_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{IC, OC}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        // Rank-0 (scalar) per-tensor zero-point.
        auto zp_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_sub = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        // Rank-0 (scalar) per-tensor scale.
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_sub, scale_const);
        auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
        auto wei_t = std::make_shared<ov::op::v1::Transpose>(wei_scale, perm);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_t, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, IC});
        auto weights_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{IC, OC}, {1});
        auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
        auto weights_t = std::make_shared<ov::op::v1::Transpose>(weights_const, perm);
        // Rank-0 scale / zp pass through apply_transpose unchanged.
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1});
        auto zp_const = ov::op::v0::Constant::create(compressed_type, ov::Shape{}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_t, bias, scale_const, zp_const);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

namespace {
struct ConvertFCToCompressedGroupSizeParams {
    ov::Shape wei_shape;
    ov::Shape scale_shape;
    ov::Shape zp_shape;
    bool transposed;
    size_t expected_group_size;
};

// Weights are [OC, IC] or grouped [OC, n_groups, group_size]; transposed: [IC, OC] or [n_groups, group_size, OC]
int64_t get_oc(const ConvertFCToCompressedGroupSizeParams& p) {
    return static_cast<int64_t>(p.transposed ? p.wei_shape.back() : p.wei_shape.front());
}

int64_t get_ic(const ConvertFCToCompressedGroupSizeParams& p) {
    return static_cast<int64_t>(ov::shape_size(p.wei_shape)) / get_oc(p);
}

std::shared_ptr<ov::Model> getModel(const ConvertFCToCompressedGroupSizeParams& p) {
    const int64_t IC = get_ic(p);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, IC});
    auto weights_const = ov::op::v0::Constant::create(ov::element::u8, p.wei_shape, {1});
    std::shared_ptr<ov::Node> weights = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
    if (!p.zp_shape.empty()) {
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, p.zp_shape, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        weights = std::make_shared<ov::op::v1::Subtract>(weights, zp_convert);
    }
    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, p.scale_shape, {1});
    weights = std::make_shared<ov::op::v1::Multiply>(weights, scale_const);
    if (p.wei_shape.size() == 3) {
        const auto target = p.transposed ? std::vector<int64_t>{IC, get_oc(p)} : std::vector<int64_t>{get_oc(p), IC};
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, target);
        weights = std::make_shared<ov::op::v1::Reshape>(weights, shape, false);
    }
    if (p.transposed) {
        auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
        weights = std::make_shared<ov::op::v1::Transpose>(weights, perm);
    }
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, weights, bias);
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getModelRef(const ConvertFCToCompressedGroupSizeParams& p) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, get_ic(p)});
    auto perm = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
    auto make_const = [&](ov::element::Type type, ov::Shape shape) -> std::shared_ptr<ov::Node> {
        if (shape.size() == 3) {
            shape = p.transposed ? ov::Shape{shape[0] * shape[1], shape[2]} : ov::Shape{shape[0], shape[1] * shape[2]};
        }
        std::shared_ptr<ov::Node> node = ov::op::v0::Constant::create(type, shape, {1});
        if (p.transposed && ov::shape_size(shape) > 1) {
            node = std::make_shared<ov::op::v1::Transpose>(node, perm);
        }
        return node;
    };
    auto weights = make_const(ov::element::u8, p.wei_shape);
    auto scale = make_const(ov::element::f32, p.scale_shape);
    auto zp = p.zp_shape.empty() ? std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0})
                                 : make_const(ov::element::u8, p.zp_shape);
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    auto fc_compressed = std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights, bias, scale, zp);
    return std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
}

class ConvertFCToCompressedGroupSize : public testing::WithParamInterface<ConvertFCToCompressedGroupSizeParams>,
                                       public TransformationTestsF {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertFCToCompressedGroupSizeParams>& info) {
        const auto shape_name = [](const ov::Shape& shape) {
            std::string name = shape.empty() ? "Scalar" : "";
            for (const auto dim : shape) {
                if (!name.empty()) {
                    name += "x";
                }
                name += std::to_string(dim);
            }
            return name;
        };
        const auto& p = info.param;
        return "Weights_" + shape_name(p.wei_shape) + "_Scale_" + shape_name(p.scale_shape) + "_ZP_" +
               (p.zp_shape.empty() ? "None" : shape_name(p.zp_shape)) + "_Transposed_" + std::to_string(p.transposed) +
               "_GroupSize_" + std::to_string(p.expected_group_size);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        model = getModel(p);
        model_ref = getModelRef(p);
        manager.register_pass<ov::pass::ConvertFullyConnectedToFullyConnectedCompressed>(
            std::vector<ov::element::Type>{ov::element::f32},
            std::vector<ov::element::Type>{ov::element::u8},
            [this](const std::shared_ptr<ov::op::internal::FullyConnected>&, size_t, size_t, size_t group_size) {
                received_group_size.push_back(group_size);
                return true;
            });
    }

    void TearDown() override {
        TransformationTestsF::TearDown();
        ASSERT_EQ(received_group_size.size(), 1u);
        EXPECT_EQ(received_group_size[0], GetParam().expected_group_size);
    }

    std::vector<size_t> received_group_size;
};

TEST_P(ConvertFCToCompressedGroupSize, PredicateReceivesICGroupSize) {}

const std::vector<ConvertFCToCompressedGroupSizeParams> group_size_params = {
    // per-tensor
    {{64, 32}, {}, {}, false, 32},
    // per-OC
    {{64, 32}, {64, 1}, {}, false, 32},
    {{32, 64}, {1, 64}, {1, 64}, true, 32},
    // grouped: group size = IC / 4
    {{64, 4, 8}, {64, 4, 1}, {}, false, 8},
    {{64, 4, 8}, {64, 4, 1}, {64, 4, 1}, false, 8},
    {{4, 8, 64}, {4, 1, 64}, {}, true, 8},
    {{4, 8, 64}, {4, 1, 64}, {4, 1, 64}, true, 8},
    // per-IC: group size = 1
    {{64, 32}, {1, 32}, {}, false, 1},
    {{64, 32}, {1, 32}, {1, 32}, false, 1},
    {{32, 64}, {32, 1}, {}, true, 1},
    {{768, 2304}, {768, 1}, {}, true, 1},
    {{32, 64}, {32, 1}, {32, 1}, true, 1},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         ConvertFCToCompressedGroupSize,
                         ::testing::ValuesIn(group_size_params),
                         ConvertFCToCompressedGroupSize::getTestCaseName);
