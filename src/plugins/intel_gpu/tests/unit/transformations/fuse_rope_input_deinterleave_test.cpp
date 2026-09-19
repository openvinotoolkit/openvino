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
};

std::shared_ptr<ov::op::internal::RoPE> make_rope(const ov::Output<ov::Node>& input,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& cos,
                                                  const std::shared_ptr<ov::op::v0::Parameter>& sin,
                                                  size_t rotary_ndims,
                                                  bool input_interleaved = false) {
    ov::op::internal::RoPE::Config config;
    config.rotary_ndims = rotary_ndims;
    config.input_interleaved = input_interleaved;
    return std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input, cos, sin}, config);
}

std::shared_ptr<ov::Model> make_deinterleave_model(RejectionCase* rejection_case = nullptr) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto cos = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto sin = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1, 128});
    auto end = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 0, std::numeric_limits<int32_t>::max()});
    const auto strides_values =
        rejection_case && *rejection_case == RejectionCase::InvalidStride ? std::vector<int32_t>{1, 1, 2, 2} : std::vector<int32_t>{1, 1, 1, 2};
    auto strides = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, strides_values);
    const std::vector<int64_t> axis_mask{1, 1, 1, 0};
    auto even = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                           ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 0, 0}),
                                                           end,
                                                           strides,
                                                           axis_mask,
                                                           axis_mask);
    auto odd = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                          ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 0, 1}),
                                                          end,
                                                          strides,
                                                          axis_mask,
                                                          axis_mask);
    const bool reversed = rejection_case && *rejection_case == RejectionCase::ReversedOrder;
    auto concat = std::make_shared<ov::op::v0::Concat>(reversed ? ov::OutputVector{odd, even} : ov::OutputVector{even, odd}, 3);
    const size_t rotary_ndims = rejection_case && *rejection_case == RejectionCase::PartialRotaryWidth ? 64 : 128;
    auto rope = make_rope(concat, cos, sin, rotary_ndims);
    return std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, cos, sin});
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
                         testing::Values(RejectionCase::InvalidStride, RejectionCase::ReversedOrder, RejectionCase::PartialRotaryWidth));

}  // namespace ov::test::intel_gpu
