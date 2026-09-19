// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/qsa.hpp"

// NOTE: like ops/qsa.cpp, QSASDPA is lowered from the base ov::op::internal::QSASDPA
// type; ov::intel_gpu::op::QSASDPA ("gpu_opset") publicly inherits from it and adds no
// new fields, so ProgramBuilder::CreateSingleLayerPrimitive's type_info->parent walk
// already dispatches gpu_opset QSASDPA nodes to a factory registered for the base type
// -- no separate exact-match registration is needed.

namespace ov::intel_gpu {

namespace {

using QSASDPAIn = ov::op::internal::QSASDPA::InputIdx;
using QSAIn = ov::op::internal::QSA::InputIdx;

// Builds a small all-zero cldnn::data constant -- used for the underlying cldnn::qsa
// primitive's QSA::InputIdx::KEY / VALUE ("current-step new tokens") slots, which are
// structurally dead when skip_kv_write=true: only Q0 (qsa_kv_cache_update) reads them,
// and Q0 is never dispatched on this path:
// the main K/V append is already done upstream by an ov::intel_gpu::op::KVCache node.
cldnn::input_info make_zero_constant(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& layer_name, const std::string& suffix,
                                     const ov::PartialShape& shape, cldnn::data_types dtype) {
    OPENVINO_ASSERT(shape.is_static(), "[GPU] QSASDPA '", layer_name, "': dummy constant '", suffix, "' requires a static shape");
    const cldnn::layout const_layout(shape, dtype, cldnn::format::get_default_format(shape.size()));
    auto mem = p.get_engine().allocate_memory(const_layout, false);
    {
        cldnn::mem_lock<uint8_t> lock{mem, p.get_engine().get_service_stream()};
        std::fill(lock.begin(), lock.end(), uint8_t{0});
    }
    const auto id = layer_name + suffix;
    p.add_primitive(*op, cldnn::data(id, mem));
    return cldnn::input_info(id);
}

// Builds an int32 cldnn::data constant holding [0, 1, ..., count-1] -- the degenerate
// (identity) block table QSASDPA's dense caches are reinterpreted through: one physical
// page per sequence, page i == sequence i.
cldnn::input_info make_iota_constant(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& layer_name, const std::string& suffix,
                                     int64_t count) {
    const cldnn::layout const_layout(ov::PartialShape{count}, cldnn::data_types::i32, cldnn::format::bfyx);
    auto mem = p.get_engine().allocate_memory(const_layout, false);
    {
        cldnn::mem_lock<int32_t> lock{mem, p.get_engine().get_service_stream()};
        for (int64_t i = 0; i < count; i++) {
            lock[i] = static_cast<int32_t>(i);
        }
    }
    const auto id = layer_name + suffix;
    p.add_primitive(*op, cldnn::data(id, mem));
    return cldnn::input_info(id);
}

void CreateQSASDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::QSASDPA>& op) {
    validate_inputs_count(op, {QSASDPAIn::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& config = op->get_config();

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::QSASDPA::validate_config(config, reason), "[GPU] Cannot lower QSASDPA op '", layer_name, "': ", reason);

    // design QSA_SDPA_DENSE: the main K/V and indexer raw-K/summary
    // caches are genuinely dynamically-shaped (e.g. the real, growing output of an
    // ov::intel_gpu::op::KVCache / QSAIndexerCache node) -- there is no requirement that
    // their sequence axis equal a fixed compile-time config.pa_block_size, and no
    // identity/degenerate page table is synthesized any more. The underlying cldnn::qsa
    // primitive is dispatched with dense_mode=true (see primitives/qsa.hpp and
    // graph/impls/cm/qsa/qsa.cpp's dense_pa_block_size()): the existing paged address
    // math is reused UNCHANGED with QSA_PA_BLOCK_SIZE derived from THIS call's actual
    // KEY_CACHE_DENSE runtime layout instead of a fixed Config value, and
    // block_indices[i] == i / block_indices_begins == identity (one degenerate page per
    // sequence, unchanged) still makes each sequence's own dense row the "whole page" --
    // only now that page's size can differ from call to call.
    const auto& key_cache_ps = op->get_input_partial_shape(QSASDPAIn::KEY_CACHE_DENSE);
    OPENVINO_ASSERT(key_cache_ps.rank().is_static() && key_cache_ps.size() == 4, "[GPU] QSASDPA '", layer_name, "': KEY_CACHE_DENSE must be rank-4");
    OPENVINO_ASSERT(key_cache_ps[0].is_static(), "[GPU] QSASDPA '", layer_name, "': KEY_CACHE_DENSE's batch dimension must be static (v1 requires a fixed"
                                                 " batch of independently-growing sequences)");
    const int64_t B = key_cache_ps[0].get_length();

    const auto& past_lens_ps = op->get_input_partial_shape(QSASDPAIn::PAST_LENS);
    OPENVINO_ASSERT(past_lens_ps.rank().is_static() && past_lens_ps[0].is_static() && past_lens_ps[0].get_length() == B,
                    "[GPU] QSASDPA '",
                    layer_name,
                    "': PAST_LENS's batch dimension must be static and equal to KEY_CACHE_DENSE's (",
                    B,
                    ")");

    const int64_t Hkv = static_cast<int64_t>(config.kv_heads_num);
    const int64_t Dh = static_cast<int64_t>(config.k_head_size);
    const int64_t Dv = static_cast<int64_t>(config.v_head_size);
    OPENVINO_ASSERT(config.compress_ratio > 0, "[GPU] QSASDPA '", layer_name, "': compress_ratio must be > 0");

    std::vector<cldnn::input_info> qsa_inputs(QSAIn::COUNT);
    qsa_inputs[QSAIn::MIXED] = inputs[QSASDPAIn::MIXED];
    qsa_inputs[QSAIn::QUERY] = inputs[QSASDPAIn::QUERY];
    qsa_inputs[QSAIn::KEY] = make_zero_constant(p, op, layer_name, "_dummy_key", ov::PartialShape{1, Hkv * Dh}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::VALUE] = make_zero_constant(p, op, layer_name, "_dummy_value", ov::PartialShape{1, Hkv * Dv}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::PAST_LENS] = inputs[QSASDPAIn::PAST_LENS];
    qsa_inputs[QSAIn::SUBSEQUENCE_BEGINS] = inputs[QSASDPAIn::SUBSEQUENCE_BEGINS];
    qsa_inputs[QSAIn::BLOCK_INDICES] = make_iota_constant(p, op, layer_name, "_block_indices", B);
    qsa_inputs[QSAIn::BLOCK_INDICES_BEGINS] = make_iota_constant(p, op, layer_name, "_block_indices_begins", B + 1);
    qsa_inputs[QSAIn::KEY_CACHE] = inputs[QSASDPAIn::KEY_CACHE_DENSE];
    qsa_inputs[QSAIn::VALUE_CACHE] = inputs[QSASDPAIn::VALUE_CACHE_DENSE];
    // Fed directly as dense [B, L, Di] tensors: for the degenerate one-page-per-sequence
    // table, a plain row-major dense tensor's byte layout is IDENTICAL to the "paged"
    // [B, 1, L, Di] layout the address math conceptually assumes (the page axis
    // contributes a zero offset for page 0 regardless of whether it is materialized), so
    // no reshape is needed -- unlike the previous (replaced) implementation, which
    // required a fixed pa_block_size-sized paged reshape.
    qsa_inputs[QSAIn::INDEXER_RAW_K_CACHE] = inputs[QSASDPAIn::INDEXER_RAW_K_CACHE_DENSE];
    qsa_inputs[QSAIn::INDEXER_SUMMARY_CACHE] = inputs[QSASDPAIn::INDEXER_SUMMARY_CACHE_DENSE];
    qsa_inputs[QSAIn::INDEXER_QK_WEIGHT] = inputs[QSASDPAIn::INDEXER_QK_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_Q_NORM_WEIGHT] = inputs[QSASDPAIn::INDEXER_Q_NORM_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_K_NORM_WEIGHT] = inputs[QSASDPAIn::INDEXER_K_NORM_WEIGHT];
    qsa_inputs[QSAIn::ROTARY_COS_SIN] = inputs[QSASDPAIn::ROTARY_COS_SIN];
    qsa_inputs[QSAIn::POSITION_IDS] = inputs[QSASDPAIn::POSITION_IDS];

    cldnn::qsa prim(layer_name, qsa_inputs, config, /*skip_kv_write=*/true, /*prepare_only=*/false, /*dense_mode=*/true);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, QSASDPA);

}  // namespace ov::intel_gpu
