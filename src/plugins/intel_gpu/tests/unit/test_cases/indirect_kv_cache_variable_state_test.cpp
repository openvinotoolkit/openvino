// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/read_value.hpp>
#include <intel_gpu/primitives/assign.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

// Helper: create a minimal network to obtain a ShapePredictor instance
cldnn::network::ptr make_dummy_network(cldnn::engine& engine) {
    cldnn::layout dummy_layout({1}, ov::element::f32, cldnn::format::bfyx);
    topology topo;
    topo.add(input_layout("input", dummy_layout));
    topo.add(read_value{"rv", {input_info("input")}, "v0", {dummy_layout}});
    topo.add(assign{"assign", {input_info("rv")}, "v0", dummy_layout});
    return get_network(engine, topo, get_test_default_config(engine), get_test_stream_ptr(), false);
}

}  // namespace

// ─── VariableStateIndirectKVCache tests ─────────────────────────────────────

class IndirectKVCacheStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = &get_test_engine();
        network_ = make_dummy_network(*engine_);
        context_ = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine_->get_device()});
        predictor_ = network_->get_shape_predictor();
    }

    cldnn::engine* engine_ = nullptr;
    cldnn::network::ptr network_;
    std::shared_ptr<RemoteContextImpl> context_;
    ShapePredictor::Ptr predictor_;

    // Typical KV cache: [batch, heads, seq_len, head_dim]
    static constexpr int64_t kBatch   = 1;
    static constexpr int64_t kHeads   = 4;
    static constexpr int64_t kSeqLen  = 32;
    static constexpr int64_t kHeadDim = 64;
    static constexpr size_t kBeamAxis   = 0;
    static constexpr size_t kConcatAxis = 2;  // sequence axis

    std::shared_ptr<VariableStateIndirectKVCache> make_kv_state(int64_t seq_len = kSeqLen) {
        cldnn::layout kv_layout({kBatch, kHeads, seq_len, kHeadDim}, ov::element::f16, cldnn::format::bfyx);
        VariableStateInfo info("kv_cache_0", kv_layout);
        return std::make_shared<VariableStateIndirectKVCache>(info, context_, predictor_, kBeamAxis, kConcatAxis);
    }
};

// Test: get_shape returns correct initial shape
TEST_F(IndirectKVCacheStateTest, GetShapeReturnsInitialShape) {
    auto state = make_kv_state();
    auto shape = state->get_shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], kBatch);
    EXPECT_EQ(shape[1], kHeads);
    EXPECT_EQ(shape[2], kSeqLen);
    EXPECT_EQ(shape[3], kHeadDim);
}

// Test: get_concat_axis returns configured axis
TEST_F(IndirectKVCacheStateTest, GetConcatAxisReturnsConfiguredValue) {
    auto state = make_kv_state();
    EXPECT_EQ(state->get_concat_axis(), kConcatAxis);
}

// Test: set_shape with same shape is a no-op
TEST_F(IndirectKVCacheStateTest, SetShapeSameShapeIsNoop) {
    auto state = make_kv_state();
    ov::Shape same_shape{kBatch, kHeads, kSeqLen, kHeadDim};
    state->set_shape(same_shape);  // should not throw
    EXPECT_EQ(state->get_shape(), same_shape);
}

// Test: set_shape trims KV cache — shape updates correctly
TEST_F(IndirectKVCacheStateTest, SetShapeTrimsKVCacheShape) {
    auto state = make_kv_state(32);
    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);
    EXPECT_EQ(state->get_shape(), trimmed);
}

// Test: set_shape adjusts padding on KV state to preserve total buffer size
TEST_F(IndirectKVCacheStateTest, SetShapeAdjustsKVPadding) {
    auto state = make_kv_state(32);

    auto layout_before = state->get_layout();
    auto pad_before = layout_before.data_padding._upper_size[kConcatAxis];

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto layout_after = state->get_layout();
    auto pad_after = layout_after.data_padding._upper_size[kConcatAxis];

    // Padding should increase by the trim delta (32 - 20 = 12)
    EXPECT_EQ(pad_after, pad_before + 12);
}

// Test: set_shape adjusts beam table state shape
TEST_F(IndirectKVCacheStateTest, SetShapeTrimsBeamTableShape) {
    auto state = make_kv_state(32);

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto bt_state = state->get_beam_table_state();
    auto bt_shape = bt_state->get_layout().get_shape();

    // Beam table shape: [batch, 1, seq_len, 1] (only beam_axis and concat_axis are non-1)
    EXPECT_EQ(bt_shape[kBeamAxis], kBatch);
    EXPECT_EQ(bt_shape[kConcatAxis], 20u);
}

// Test: set_shape adjusts beam table padding
TEST_F(IndirectKVCacheStateTest, SetShapeAdjustsBeamTablePadding) {
    auto state = make_kv_state(32);

    auto bt_state = state->get_beam_table_state();
    auto bt_pad_before = bt_state->get_layout().data_padding._upper_size[kConcatAxis];

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto bt_pad_after = bt_state->get_layout().data_padding._upper_size[kConcatAxis];
    EXPECT_EQ(bt_pad_after, bt_pad_before + 12);
}

// Test: multiple sequential trims accumulate padding correctly
TEST_F(IndirectKVCacheStateTest, MultipleTrimsAccumulatePadding) {
    auto state = make_kv_state(32);
    auto pad_initial = state->get_layout().data_padding._upper_size[kConcatAxis];

    // Trim 32 -> 28
    state->set_shape(ov::Shape{kBatch, kHeads, 28, kHeadDim});
    EXPECT_EQ(state->get_shape()[kConcatAxis], 28u);

    // Trim 28 -> 20
    state->set_shape(ov::Shape{kBatch, kHeads, 20, kHeadDim});
    EXPECT_EQ(state->get_shape()[kConcatAxis], 20u);

    // Total padding increase = 32 - 20 = 12
    auto pad_final = state->get_layout().data_padding._upper_size[kConcatAxis];
    EXPECT_EQ(pad_final, pad_initial + 12);
}

// Test: trim to 1 token (minimal sequence length)
TEST_F(IndirectKVCacheStateTest, TrimToMinimalSequenceLength) {
    auto state = make_kv_state(32);
    ov::Shape minimal{kBatch, kHeads, 1, kHeadDim};
    state->set_shape(minimal);
    EXPECT_EQ(state->get_shape()[kConcatAxis], 1u);

    auto pad = state->get_layout().data_padding._upper_size[kConcatAxis];
    EXPECT_EQ(pad, 31);  // 32 - 1 = 31
}

// ─── VariableStateIndirectKVCacheCompressed tests ───────────────────────────

class IndirectKVCacheCompressedStateTest : public IndirectKVCacheStateTest {
protected:
    static constexpr size_t kScaleSeqAxis = 2;

    std::shared_ptr<VariableStateIndirectKVCacheCompressed> make_compressed_state(
            int64_t seq_len = kSeqLen, bool with_zp = false) {
        cldnn::layout kv_layout({kBatch, kHeads, seq_len, kHeadDim}, ov::element::u8, cldnn::format::bfyx);
        cldnn::layout scale_layout({kBatch, kHeads, seq_len, kHeadDim}, ov::element::f16, cldnn::format::bfyx);

        VariableStateInfo info("kv_cache_compressed_0", kv_layout);

        std::vector<cldnn::layout> output_layouts;
        output_layouts.push_back(kv_layout);     // [0] = KV data layout (used by base class)
        output_layouts.push_back(scale_layout);   // [1] = compression scale layout

        if (with_zp) {
            cldnn::layout zp_layout({kBatch, kHeads, seq_len, kHeadDim}, ov::element::u8, cldnn::format::bfyx);  // NOLINT
            output_layouts.push_back(zp_layout);  // [2] = compression zero-point layout
        }

        return std::make_shared<VariableStateIndirectKVCacheCompressed>(
            info, context_, predictor_, output_layouts, kBeamAxis, kConcatAxis, with_zp);
    }
};

// Test: compressed get_shape works (delegates to base)
TEST_F(IndirectKVCacheCompressedStateTest, GetShapeReturnsCorrectShape) {
    auto state = make_compressed_state();
    auto shape = state->get_shape();
    EXPECT_EQ(shape[kConcatAxis], kSeqLen);
}

// Test: compressed set_shape trims KV and scale states
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeTrimsKVAndScale) {
    auto state = make_compressed_state(32);

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    // KV shape trimmed
    EXPECT_EQ(state->get_shape()[kConcatAxis], 20u);

    // Scale state trimmed on scale_seq_axis (axis 2)
    auto scale_state = state->get_compression_scale_state();
    auto scale_shape = scale_state->get_layout().get_shape();
    EXPECT_EQ(scale_shape[kScaleSeqAxis], 20u);
}

// Test: compressed set_shape adjusts scale padding
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeAdjustsScalePadding) {
    auto state = make_compressed_state(32);

    auto scale_state = state->get_compression_scale_state();
    auto scale_pad_before = scale_state->get_layout().data_padding._upper_size[kScaleSeqAxis];

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto scale_pad_after = scale_state->get_layout().data_padding._upper_size[kScaleSeqAxis];
    EXPECT_EQ(scale_pad_after, scale_pad_before + 12);
}

// Test: compressed set_shape with zero-points trims zp state
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeTrimsZeroPointState) {
    auto state = make_compressed_state(32, /*with_zp=*/true);
    ASSERT_TRUE(state->has_zp_state());

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto zp_state = state->get_compression_zp_state();
    auto zp_shape = zp_state->get_layout().get_shape();
    EXPECT_EQ(zp_shape[kScaleSeqAxis], 20u);
}

// Test: compressed set_shape with zero-points adjusts zp padding
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeAdjustsZPPadding) {
    auto state = make_compressed_state(32, /*with_zp=*/true);

    auto zp_state = state->get_compression_zp_state();
    auto zp_pad_before = zp_state->get_layout().data_padding._upper_size[kScaleSeqAxis];

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    auto zp_pad_after = zp_state->get_layout().data_padding._upper_size[kScaleSeqAxis];
    EXPECT_EQ(zp_pad_after, zp_pad_before + 12);
}

// Test: compressed set_shape without zero-points state
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeWithoutZPState) {
    auto state = make_compressed_state(32, /*with_zp=*/false);
    ASSERT_FALSE(state->has_zp_state());

    ov::Shape trimmed{kBatch, kHeads, 20, kHeadDim};
    state->set_shape(trimmed);

    // KV + scale trimmed, no crash
    EXPECT_EQ(state->get_shape()[kConcatAxis], 20u);
    auto scale_shape = state->get_compression_scale_state()->get_layout().get_shape();
    EXPECT_EQ(scale_shape[kScaleSeqAxis], 20u);
}

// Test: compressed same-shape set_shape is a no-op
TEST_F(IndirectKVCacheCompressedStateTest, SetShapeSameShapeIsNoop) {
    auto state = make_compressed_state(32);
    ov::Shape same{kBatch, kHeads, 32, kHeadDim};

    auto scale_pad_before = state->get_compression_scale_state()->get_layout().data_padding._upper_size[kScaleSeqAxis];
    state->set_shape(same);
    auto scale_pad_after = state->get_compression_scale_state()->get_layout().data_padding._upper_size[kScaleSeqAxis];

    EXPECT_EQ(scale_pad_before, scale_pad_after);
}

// Test: compressed multiple trims accumulate correctly for all states
TEST_F(IndirectKVCacheCompressedStateTest, MultipleTrimsAccumulateForAllStates) {
    auto state = make_compressed_state(32, /*with_zp=*/true);

    state->set_shape(ov::Shape{kBatch, kHeads, 28, kHeadDim});
    state->set_shape(ov::Shape{kBatch, kHeads, 16, kHeadDim});

    // KV
    EXPECT_EQ(state->get_shape()[kConcatAxis], 16u);
    EXPECT_EQ(state->get_layout().data_padding._upper_size[kConcatAxis], 16);  // 32 - 16

    // Scale
    auto scale_shape = state->get_compression_scale_state()->get_layout().get_shape();
    EXPECT_EQ(scale_shape[kScaleSeqAxis], 16u);
    EXPECT_EQ(state->get_compression_scale_state()->get_layout().data_padding._upper_size[kScaleSeqAxis], 16);

    // ZP
    auto zp_shape = state->get_compression_zp_state()->get_layout().get_shape();
    EXPECT_EQ(zp_shape[kScaleSeqAxis], 16u);
    EXPECT_EQ(state->get_compression_zp_state()->get_layout().data_padding._upper_size[kScaleSeqAxis], 16);

    // Beam table
    auto bt_shape = state->get_beam_table_state()->get_layout().get_shape();
    EXPECT_EQ(bt_shape[kConcatAxis], 16u);
}

// Test: compressed get_state throws (by design)
TEST_F(IndirectKVCacheCompressedStateTest, GetStateThrows) {
    auto state = make_compressed_state();
    EXPECT_THROW(state->get_state(), ov::Exception);
}

// Test: compressed set_state throws (by design)
TEST_F(IndirectKVCacheCompressedStateTest, SetStateThrows) {
    auto state = make_compressed_state();
    auto tensor = ov::SoPtr<ov::ITensor>();
    EXPECT_THROW(state->set_state(tensor), ov::Exception);
}
