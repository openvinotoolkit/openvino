// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_rope_input_deinterleave.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::test::intel_gpu {
namespace {

enum class RejectionCase {
    InvalidStride,
    ReversedOrder,
    PartialRotaryWidth,
    DifferentInputs,
    NonPlainConfig,
    GatherConfig,
    ExtraConcatConsumer,
    ExtraSliceConsumer,
};

std::shared_ptr<ov::op::internal::RoPE> make_rope(const ov::Output<ov::Node>& input,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& cos,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& sin,
                                                  size_t rotary_ndims,
                                                  bool input_interleaved = false,
                                                  bool input_trans0213 = false,
                                                  int gather_position_arg_id = 0) {
    ov::op::internal::RoPE::Config config;
    config.rotary_ndims = rotary_ndims;
    config.input_interleaved = input_interleaved;
    config.input_trans0213 = input_trans0213;
    config.gather_position_arg_id = gather_position_arg_id;
    return std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input, cos, sin}, config);
}

std::shared_ptr<ov::Model> make_deinterleave_model(RejectionCase* rejection_case = nullptr, bool negative_axis = false, bool i64_indices = false) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto cos = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto sin = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    const auto index_type = i64_indices ? ov::element::i64 : ov::element::i32;
    const int64_t end_value = i64_indices ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int32_t>::max();
    auto end = ov::op::v0::Constant::create(index_type, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, end_value});
    const auto strides_values =
        rejection_case && *rejection_case == RejectionCase::InvalidStride ? std::vector<int64_t>{1, 1, 2, 2} : std::vector<int64_t>{1, 1, 1, 2};
    auto strides = ov::op::v0::Constant::create(index_type, ov::Shape{4}, strides_values);
    const std::vector<int64_t> axis_mask{1, 1, 1, 0};
    auto even = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                           ov::op::v0::Constant::create(index_type, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0}),
                                                           end,
                                                           strides,
                                                           axis_mask,
                                                           axis_mask);
    ov::ParameterVector parameters{input, cos, sin};
    ov::Output<ov::Node> odd_input = input;
    if (rejection_case && *rejection_case == RejectionCase::DifferentInputs) {
        auto other_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
        odd_input = other_input;
        parameters.push_back(other_input);
    }
    auto odd = std::make_shared<ov::op::v1::StridedSlice>(odd_input,
                                                          ov::op::v0::Constant::create(index_type, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 1}),
                                                          end,
                                                          strides,
                                                          axis_mask,
                                                          axis_mask);
    const bool reversed = rejection_case && *rejection_case == RejectionCase::ReversedOrder;
    auto concat = std::make_shared<ov::op::v0::Concat>(reversed ? ov::OutputVector{odd, even} : ov::OutputVector{even, odd}, negative_axis ? -1 : 3);
    const size_t rotary_ndims = rejection_case && *rejection_case == RejectionCase::PartialRotaryWidth ? 64 : 128;
    const bool non_plain_config = rejection_case && *rejection_case == RejectionCase::NonPlainConfig;
    const int gather_position_arg_id = rejection_case && *rejection_case == RejectionCase::GatherConfig ? 3 : 0;
    auto rope = make_rope(concat, cos, sin, rotary_ndims, false, non_plain_config, gather_position_arg_id);
    ov::OutputVector outputs{rope};
    if (rejection_case && *rejection_case == RejectionCase::ExtraConcatConsumer) {
        outputs.push_back(concat);
    } else if (rejection_case && *rejection_case == RejectionCase::ExtraSliceConsumer) {
        outputs.push_back(even);
    }
    return std::make_shared<ov::Model>(outputs, parameters);
}

}  // namespace

TEST_F(TransformationTestsF, FuseRoPEInputDeinterleave) {
    model = make_deinterleave_model();
    manager.register_pass<ov::intel_gpu::FuseRoPEInputDeinterleave>();

    auto input_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto cos_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto sin_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto rope_ref = make_rope(input_ref, cos_ref, sin_ref, 128, true);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope_ref}, ov::ParameterVector{input_ref, cos_ref, sin_ref});
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, FuseRoPEInputDeinterleaveNegativeAxisI64Max) {
    model = make_deinterleave_model(nullptr, true, true);
    manager.register_pass<ov::intel_gpu::FuseRoPEInputDeinterleave>();

    auto input_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto cos_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto sin_ref = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto rope_ref = make_rope(input_ref, cos_ref, sin_ref, 128, true);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope_ref}, ov::ParameterVector{input_ref, cos_ref, sin_ref});
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

class FuseRoPEInputDeinterleaveNoFusionTest : public TransformationTestsF, public testing::WithParamInterface<RejectionCase> {};

TEST_P(FuseRoPEInputDeinterleaveNoFusionTest, RejectsInexactPattern) {
    auto rejection_case = GetParam();
    model = make_deinterleave_model(&rejection_case);
    model_ref = model->clone();
    manager.register_pass<ov::intel_gpu::FuseRoPEInputDeinterleave>();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseRoPEInputDeinterleave,
                         FuseRoPEInputDeinterleaveNoFusionTest,
                         testing::Values(RejectionCase::InvalidStride,
                                         RejectionCase::ReversedOrder,
                                         RejectionCase::PartialRotaryWidth,
                                         RejectionCase::DifferentInputs,
                                         RejectionCase::NonPlainConfig,
                                         RejectionCase::GatherConfig,
                                         RejectionCase::ExtraConcatConsumer,
                                         RejectionCase::ExtraSliceConsumer));

}  // namespace ov::test::intel_gpu
