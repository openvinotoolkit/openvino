// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "kv_cache_sliding_window_manager.hpp"
#include "llm_compiled_model_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::test::npuw {

namespace {

namespace uu = ov::npuw::util;

ov::SoPtr<ov::ITensor> make_mask_tensor(uint32_t rows,
                                        uint32_t cols,
                                        float init_value,
                                        const ov::element::Type& element_type = ov::element::f32) {
    auto mask = ov::get_tensor_impl(ov::Tensor(element_type, ov::Shape{1u, rows, cols}));
    if (element_type == ov::element::f16) {
        std::fill(mask->data<ov::float16>(), mask->data<ov::float16>() + mask->get_size(), ov::float16(init_value));
    } else {
        std::fill(mask->data<float>(), mask->data<float>() + mask->get_size(), init_value);
    }
    return mask;
}

float mask_at(const ov::SoPtr<ov::ITensor>& mask, uint32_t row, uint32_t col) {
    const auto& shape = mask->get_shape();
    const auto rows = static_cast<uint32_t>(shape[shape.size() - 2]);
    const auto cols = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(row < rows && col < cols, "mask_at index out of range");
    const size_t index = static_cast<size_t>(row) * cols + col;
    if (mask->get_element_type() == ov::element::f16) {
        return static_cast<float>(mask->data<ov::float16>()[index]);
    }
    return mask->data<float>()[index];
}

// Builds a standalone Parameter port. The returned Output keeps the Parameter node alive
// (Output owns a shared_ptr to its node), so no separate storage is needed by callers.
ov::Output<const ov::Node> make_param_port(const ov::element::Type& type, const ov::Shape& shape) {
    return std::make_shared<ov::op::v0::Parameter>(type, shape)->output(0);
}

// Minimal fake IAsyncInferRequest for testing fill_sliding_window_attention_mask, which only
// calls request->get_tensor(port). Tensors are bound to ports ahead of time via bind().
class FakeAsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    FakeAsyncInferRequest() : ov::IAsyncInferRequest(nullptr, nullptr, nullptr) {}

    void bind(const ov::Output<const ov::Node>& port, ov::SoPtr<ov::ITensor> tensor) {
        m_tensors[port.get_node()] = std::move(tensor);
    }

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        const auto it = m_tensors.find(port.get_node());
        OPENVINO_ASSERT(it != m_tensors.end(), "FakeAsyncInferRequest: no tensor bound for the requested port");
        return it->second;
    }

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        m_tensors[port.get_node()] = tensor;
    }

private:
    std::unordered_map<const ov::Node*, ov::SoPtr<ov::ITensor>> m_tensors;
};

}  // namespace

class FillCausalSlidingWindowMaskTest : public ::testing::Test {};

TEST_F(FillCausalSlidingWindowMaskTest, UnsaturatedPastBuildsExpectedMask) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/2u,
                                        /*num_new_tokens=*/2u,
                                        /*window_size=*/3u);

    // row=2: q=2, past abs=[0,1] visible; only diagonal local key is visible.
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), kMasked);

    // row=3: abs=0 is out-of-window, abs=1 is still visible.
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);
}

TEST_F(FillCausalSlidingWindowMaskTest, SaturatedPastUsesCircularSlotMapping) {
    auto mask = make_mask_tensor(/*rows=*/2u, /*cols=*/6u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/6u,
                                        /*num_new_tokens=*/2u,
                                        /*window_size=*/3u);

    // past_width=4, r=2 => slot->abs=[4,5,2,3]; row=0(q=6): 4/5 visible, 2/3 masked.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), kMasked);

    // row=1(q=7): abs 5 visible, abs 4 masked.
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);
}

TEST_F(FillCausalSlidingWindowMaskTest, ZeroPastWidthFallsBackToCurrentChunkSlidingWindowCausalMask) {
    // rows == cols => past_width == 0, so mask must be built from current chunk only.
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/4u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/0u,
                                        /*num_new_tokens=*/4u,
                                        /*window_size=*/2u);

    // Expected visible local columns per row for window=2:
    // row0 -> [0]
    // row1 -> [0,1]
    // row2 -> [1,2]
    // row3 -> [2,3]
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 2u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 3u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 3u), 0.f);
}

TEST_F(FillCausalSlidingWindowMaskTest, UnsaturatedPastBuildsExpectedMaskF16) {
    // Same scenario as UnsaturatedPastBuildsExpectedMask, but on an f16 mask tensor to cover
    // the fill_causal_sliding_window_mask_typed<ov::float16> instantiation.
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/777.f, ov::element::f16);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/2u,
                                        /*num_new_tokens=*/2u,
                                        /*window_size=*/3u);

    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);
}

TEST_F(FillCausalSlidingWindowMaskTest, RowPadRowsDoNotAffectRealRowVisibility) {
    // num_new_tokens(2) < row_dim(4) => row_pad=2: the first 2 rows are padding and the real
    // chunk occupies rows [2,4). past_width=4, stored_tokens=3 (unsaturated).
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/3u,
                                        /*num_new_tokens=*/2u,
                                        /*window_size=*/3u);

    // row=2 is the real row 0: abs=3, visible past slots [1,2], present local_c=2 (col 6).
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 4u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 5u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), kMasked);

    // row=3 is the real row 1: abs=4, visible past slot [2], present local_c in [2,3] (cols 6,7).
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);
}

TEST_F(FillCausalSlidingWindowMaskTest, SaturatedPastWithExactWrapUsesSingleSegment) {
    // stored_tokens(8) is an exact multiple of past_width(4) => wrap_slot=0, so the saturated
    // ring resolves to a single contiguous segment (segment2 stays empty).
    auto mask = make_mask_tensor(/*rows=*/2u, /*cols=*/6u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_window_mask(mask,
                                        /*num_stored_tokens=*/8u,
                                        /*num_new_tokens=*/2u,
                                        /*window_size=*/3u);

    // row=0(abs=8): visible past slots [2,3] -> cols 2,3; present local_c=0 -> col 4.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 3u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), kMasked);

    // row=1(abs=9): visible past slot [3] -> col 3; present local_c in [0,1] -> cols 4,5.
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 3u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);
}

class OverlayVisionBidirectionalMaskTest : public ::testing::Test {};

TEST_F(OverlayVisionBidirectionalMaskTest, SingleVisionRunUnmasksRunBlock) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-7.f);
    const std::vector<int64_t> token_types = {0, 1, 1, 0};

    uu::overlay_vision_bidirectional_mask(mask, token_types.data(), static_cast<uint32_t>(token_types.size()));

    // run [1,3): rows 1..2 and cols 5..6 are unmasked.
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);

    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), -7.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), -7.f);
}

TEST_F(OverlayVisionBidirectionalMaskTest, DisjointVisionRunsAreHandledSeparately) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-9.f);
    const std::vector<int64_t> token_types = {1, 1, 0, 1};

    uu::overlay_vision_bidirectional_mask(mask, token_types.data(), static_cast<uint32_t>(token_types.size()));

    // run [0,2): rows 0..1 and cols 4..5.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);

    // run [3,4): row 3 and col 7.
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);

    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), -9.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), -9.f);
}

TEST_F(OverlayVisionBidirectionalMaskTest, ZeroNewTokensIsNoOp) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-3.f);
    const std::vector<int64_t> token_types = {1, 1, 0};

    uu::overlay_vision_bidirectional_mask(mask, token_types.data(), /*num_new_tokens=*/0u);

    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), -3.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), -3.f);
}

TEST_F(OverlayVisionBidirectionalMaskTest, NullTokenTypeIdsThrows) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-3.f);

    EXPECT_THROW(uu::overlay_vision_bidirectional_mask(mask, nullptr, /*num_new_tokens=*/2u), ov::Exception);
}

class FillSlidingWindowAttentionMaskTest : public ::testing::Test {};

TEST_F(FillSlidingWindowAttentionMaskTest, MaskPortMissingIsNoOp) {
    // Non-SWA models don't expose the sliding_window_attention_mask input: the function must
    // return without touching the (empty) request at all.
    auto fake_request = std::make_shared<FakeAsyncInferRequest>();
    std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports;

    EXPECT_NO_THROW(uu::fill_sliding_window_attention_mask(fake_request,
                                                           in_ports,
                                                           /*num_stored_tokens=*/3u,
                                                           /*num_new_tokens=*/3u,
                                                           /*window_size=*/4u));
}

TEST_F(FillSlidingWindowAttentionMaskTest, TokenTypeIdsPortMissingSkipsOverlay) {
    // Same numbers as the causal half of the combined-mask example documented on
    // fill_sliding_window_attention_mask, but without a token_type_ids port: only the causal
    // mask must be filled, no vision overlay applied.
    auto mask = make_mask_tensor(/*rows=*/3u, /*cols=*/7u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    auto fake_request = std::make_shared<FakeAsyncInferRequest>();
    auto mask_port = make_param_port(ov::element::f32, ov::Shape{1u, 3u, 7u});
    fake_request->bind(mask_port, mask);

    std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports;
    in_ports[uu::kSlidingWindowAttentionMaskParamName] = mask_port;

    uu::fill_sliding_window_attention_mask(fake_request,
                                           in_ports,
                                           /*num_stored_tokens=*/3u,
                                           /*num_new_tokens=*/3u,
                                           /*window_size=*/4u);

    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    // Would be unmasked by the vision overlay if a token_type_ids port had been provided.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), kMasked);
}

TEST_F(FillSlidingWindowAttentionMaskTest, CombinesCausalMaskAndVisionOverlay) {
    // Reproduces the worked example from fill_sliding_window_attention_mask's doc comment:
    // past_width=4, row_dim=3, stored_tokens=3, window_size=4, token_type_ids=[V,V,T].
    auto mask = make_mask_tensor(/*rows=*/3u, /*cols=*/7u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    auto fake_request = std::make_shared<FakeAsyncInferRequest>();
    auto mask_port = make_param_port(ov::element::f32, ov::Shape{1u, 3u, 7u});
    fake_request->bind(mask_port, mask);

    const std::vector<int64_t> token_types = {1, 1, 0};  // V, V, T
    auto token_type_ids = ov::get_tensor_impl(ov::Tensor(ov::element::i64, ov::Shape{token_types.size()}));
    std::copy(token_types.begin(), token_types.end(), token_type_ids->data<int64_t>());
    auto token_type_ids_port = make_param_port(ov::element::i64, ov::Shape{token_types.size()});
    fake_request->bind(token_type_ids_port, token_type_ids);

    std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports;
    in_ports[uu::kSlidingWindowAttentionMaskParamName] = mask_port;
    in_ports[uu::kTokenTypeIdsParamName] = token_type_ids_port;

    uu::fill_sliding_window_attention_mask(fake_request,
                                           in_ports,
                                           /*num_stored_tokens=*/3u,
                                           /*num_new_tokens=*/3u,
                                           /*window_size=*/4u);

    // row 0: vision overlay forces col 5 to attend on top of the causal mask.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 6u), kMasked);

    // row 2: outside the [0,2) vision run, unaffected by the overlay.
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);
}

TEST_F(FillSlidingWindowAttentionMaskTest, TokenTypeIdsSizeMismatchThrows) {
    auto mask = make_mask_tensor(/*rows=*/3u, /*cols=*/7u, /*init_value=*/777.f);

    auto fake_request = std::make_shared<FakeAsyncInferRequest>();
    auto mask_port = make_param_port(ov::element::f32, ov::Shape{1u, 3u, 7u});
    fake_request->bind(mask_port, mask);

    // token_type_ids tensor (size 2) is smaller than num_new_tokens (3).
    auto token_type_ids = ov::get_tensor_impl(ov::Tensor(ov::element::i64, ov::Shape{2u}));
    auto token_type_ids_port = make_param_port(ov::element::i64, ov::Shape{2u});
    fake_request->bind(token_type_ids_port, token_type_ids);

    std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports;
    in_ports[uu::kSlidingWindowAttentionMaskParamName] = mask_port;
    in_ports[uu::kTokenTypeIdsParamName] = token_type_ids_port;

    EXPECT_THROW(uu::fill_sliding_window_attention_mask(fake_request,
                                                        in_ports,
                                                        /*num_stored_tokens=*/3u,
                                                        /*num_new_tokens=*/3u,
                                                        /*window_size=*/4u),
                 ov::Exception);
}

}  // namespace ov::test::npuw
