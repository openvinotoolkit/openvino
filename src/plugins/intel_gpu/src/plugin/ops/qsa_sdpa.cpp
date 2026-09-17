// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/qsa.hpp"
#include "intel_gpu/primitives/reshape.hpp"

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

// Reinterprets a dense [B, S, Di] indexer cache tensor as the paged [B, 1, S, Di] shape
// cldnn::qsa expects (pure relabeling: same element count, same memory, no data
// movement).
cldnn::input_info make_dense_to_paged_reshape(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& layer_name,
                                              const std::string& suffix, const cldnn::input_info& input, const ov::PartialShape& target_shape) {
    OPENVINO_ASSERT(target_shape.is_static(), "[GPU] QSASDPA '", layer_name, "': reshape '", suffix, "' requires a static target shape");
    std::vector<int64_t> pattern;
    pattern.reserve(target_shape.size());
    for (const auto& d : target_shape) {
        pattern.push_back(d.get_length());
    }
    const auto id = layer_name + suffix;
    p.add_primitive(*op, cldnn::reshape(id, input, /*special_zero=*/false, pattern, target_shape));
    return cldnn::input_info(id);
}

void CreateQSASDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::QSASDPA>& op) {
    validate_inputs_count(op, {QSASDPAIn::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& config = op->get_config();

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::QSASDPA::validate_config(config, reason), "[GPU] Cannot lower QSASDPA op '", layer_name, "': ", reason);

    // v1 requires a fully static, compile-time-known degenerate page table. Rationale:
    // pa_block_size is baked into the underlying CM kernels as a JIT constant (see
    // graph/impls/cm/qsa/qsa.cpp's "QSA_PA_BLOCK_SIZE" jit define), so the "one page per
    // sequence, page size == max sequence length" degenerate scheme this op lowers to can
    // only work if both the number of sequences (B) and the page size (S) are fixed at
    // graph-construction time. This is the concrete, enforced form of "dynamic B is
    // unsupported", extended to also require a fixed
    // max-sequence-length capacity (section 8): callers must size KEY_CACHE_DENSE /
    // VALUE_CACHE_DENSE / INDEXER_RAW_K_CACHE_DENSE / INDEXER_SUMMARY_CACHE_DENSE's S axis
    // to config.pa_block_size up front (a "reserve full capacity" strategy, not a
    // dynamically-growing-exactly-to-S one -- see the implementation record for why the
    // latter would require far more invasive graph surgery).
    const auto& key_cache_ps = op->get_input_partial_shape(QSASDPAIn::KEY_CACHE_DENSE);
    OPENVINO_ASSERT(key_cache_ps.rank().is_static() && key_cache_ps.size() == 4, "[GPU] QSASDPA '", layer_name, "': KEY_CACHE_DENSE must be rank-4");
    OPENVINO_ASSERT(key_cache_ps.is_static(),
                    "[GPU] QSASDPA '",
                    layer_name,
                    "': KEY_CACHE_DENSE must have a fully static shape in v1 (dynamic B / unbounded max sequence "
                    "length are not supported)");
    const int64_t B = key_cache_ps[0].get_length();
    const int64_t S = key_cache_ps[2].get_length();
    OPENVINO_ASSERT(static_cast<size_t>(S) == config.pa_block_size,
                    "[GPU] QSASDPA '",
                    layer_name,
                    "': KEY_CACHE_DENSE's sequence axis (",
                    S,
                    ") must equal config.pa_block_size (",
                    config.pa_block_size,
                    "), the fixed max-sequence-length capacity this op's degenerate paging reserves upfront");

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
    const int64_t Di = static_cast<int64_t>(config.indexer_head_dim);
    const int64_t r = static_cast<int64_t>(config.compress_ratio);
    OPENVINO_ASSERT(r > 0, "[GPU] QSASDPA '", layer_name, "': compress_ratio must be > 0");
    const int64_t summaries = S / r;

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
    qsa_inputs[QSAIn::INDEXER_RAW_K_CACHE] =
        make_dense_to_paged_reshape(p, op, layer_name, "_raw_k_paged", inputs[QSASDPAIn::INDEXER_RAW_K_CACHE_DENSE], ov::PartialShape{B, 1, S, Di});
    qsa_inputs[QSAIn::INDEXER_SUMMARY_CACHE] = make_dense_to_paged_reshape(p, op, layer_name, "_summary_paged", inputs[QSASDPAIn::INDEXER_SUMMARY_CACHE_DENSE],
                                                                          ov::PartialShape{B, 1, summaries, Di});
    qsa_inputs[QSAIn::INDEXER_QK_WEIGHT] = inputs[QSASDPAIn::INDEXER_QK_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_Q_NORM_WEIGHT] = inputs[QSASDPAIn::INDEXER_Q_NORM_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_K_NORM_WEIGHT] = inputs[QSASDPAIn::INDEXER_K_NORM_WEIGHT];
    qsa_inputs[QSAIn::ROTARY_COS_SIN] = inputs[QSASDPAIn::ROTARY_COS_SIN];
    qsa_inputs[QSAIn::POSITION_IDS] = inputs[QSASDPAIn::POSITION_IDS];

    cldnn::qsa prim(layer_name, qsa_inputs, config, /*skip_kv_write=*/true, /*prepare_only=*/false);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, QSASDPA);

}  // namespace ov::intel_gpu
