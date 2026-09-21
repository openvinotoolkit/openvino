// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa_indexer_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/qsa.hpp"
#include "intel_gpu/primitives/read_value.hpp"

// NOTE: QSAIndexerCache has no ov::op::internal counterpart (like ov::intel_gpu::op::
// KVCache, and unlike QSA/QSASDPA/GatedResidual): it is produced by the GPU graph-fusion
// path and is not constructed directly by a model author, so it does not
// need a dev_api-exposed base class. REGISTER_FACTORY_IMPL(internal, X) hardcodes the C++
// type as ov::op::internal::X, so -- exactly like ops/kv_cache.cpp does for KVCache -- an
// alias is declared so the macro resolves to the real (gpu_opset-only) class. This is safe
// here specifically because no real ov::op::internal::QSAIndexerCache class exists to
// conflict with it (see the QSA/GatedResidual dev_api exposure memory for the failure mode
// this would otherwise hit).
namespace ov {
namespace op {
namespace internal {
using QSAIndexerCache = ov::intel_gpu::op::QSAIndexerCache;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

namespace {

using QSAIn = ov::intel_gpu::op::QSA::InputIdx;

// See ops/qsa_sdpa.cpp for the identical helpers (kept as separate, self-contained
// copies -- each ops/*.cpp file in this directory is independent, no shared internal
// lowering-helpers header exists yet).

cldnn::input_info make_zero_constant(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& layer_name, const std::string& suffix,
                                     const ov::PartialShape& shape, cldnn::data_types dtype) {
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

// QSAIndexerCache is a pure state DECLARATION: it has zero real data inputs and must not compute or
// write any cache content. Its lowering therefore dispatches ZERO kernel stages -- it
// reads the two Variables' CURRENT content via ordinary `cldnn::read_value` primitives,
// then feeds them through `cldnn::qsa` with `state_only=true`, which only ALIASES its two
// declared outputs to those same two inputs (see impls/cm/qsa/qsa.cpp's QSACmImpl ctor
// and execute()). This reuses `cldnn::qsa`'s already-proven multi-output declaration (the
// SAME mechanism `prepare_only` uses) so that ProgramBuilder's generic GetInputInfo()
// output-port resolution (see program_builder.cpp) works for downstream consumers (e.g.
// QSASDPA) exactly like any other multi-output node -- a bare `cldnn::read_value` cannot
// do this itself since its runtime only ever supports one real output
// (read_value_inst::calc_output_layout() always returns output_layouts[0]).
//
// The reserve/grow/Q1-write/publish responsibility that used to live here now belongs
// entirely to the `qsa_inst` that executes the corresponding QSASDPA node (see
// ops/qsa_sdpa.cpp::CreateQSASDPAOp), which locates these SAME two Variables via this
// upstream node and issues the `Assign`s that actually persist the grown state.
//
// The eleven other QSAIn slots (MIXED/QUERY/KEY/VALUE/PAST_LENS/SUBSEQUENCE_BEGINS/
// BLOCK_INDICES/BLOCK_INDICES_BEGINS/weights/ROTARY_COS_SIN/POSITION_IDS) are
// structurally dead in state_only mode (no kernel ever reads them, and
// get_internal_buffer_descs() skips sizing entirely for state_only -- see
// impls/cm/qsa/qsa.cpp); minimal fixed-size dummy constants satisfy the primitive's fixed
// 17-input arity without depending on any real graph tensor.
void CreateQSAIndexerCacheOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::QSAIndexerCache>& op) {
    validate_inputs_count(op, {0});
    const auto layer_name = layer_type_name_ID(op);
    const auto& config = op->get_config();

    auto raw_k_variable = op->get_variable();
    auto summary_variable = op->get_summary_variable();
    OPENVINO_ASSERT(raw_k_variable && summary_variable, "[GPU] QSAIndexerCache '", layer_name, "' requires both a raw-K and a summary Variable");

    const int64_t Di = static_cast<int64_t>(config.indexer_head_dim);

    // Keep the indexer state at a fixed capacity. QSA_PA_BLOCK_SIZE is a JIT constant;
    // growing this layout on every decode step would create a new kernel specialization.
    const auto& raw_k_shape = raw_k_variable->get_info().data_shape;
    OPENVINO_ASSERT(raw_k_shape.rank().is_static() && raw_k_shape.size() == 3 && raw_k_shape[0].is_static(),
                    "[GPU] QSAIndexerCache '", layer_name, "': raw-K Variable must be a rank-3 [B, ?, Di] shape with a static batch dimension");
    OPENVINO_ASSERT(raw_k_shape[2].is_static() && raw_k_shape[2].get_length() == Di,
                    "[GPU] QSAIndexerCache '", layer_name, "': raw-K Variable's last dim must equal config.indexer_head_dim");
    const int64_t B = raw_k_shape[0].get_length();

    const auto& summary_shape = summary_variable->get_info().data_shape;
    OPENVINO_ASSERT(summary_shape.rank().is_static() && summary_shape.size() == 3 && summary_shape[0].is_static() &&
                        summary_shape[0].get_length() == B && summary_shape[2].is_static() && summary_shape[2].get_length() == Di,
                    "[GPU] QSAIndexerCache '", layer_name, "': summary Variable must be a rank-3 [", B,
                    ", ?, Di] shape consistent with the raw-K Variable and config");

    const int64_t Hkv = static_cast<int64_t>(config.kv_heads_num);
    const int64_t Dh = static_cast<int64_t>(config.k_head_size);
    const int64_t Dv = static_cast<int64_t>(config.v_head_size);
    const int64_t Hidx = static_cast<int64_t>(config.indexer_heads);
    const int64_t D = static_cast<int64_t>(config.hidden_size);

    // ReadValue (0 inputs): returns each Variable's CURRENT valid-length content. The
    // declared layout's sequence axis reuses the Variable's OWN declared dimension as-is
    // (static, bounded-dynamic, or fully dynamic -- whatever the caller declared it as;
    // see qsa_indexer_cache.hpp's class comment) rather than a fixed pa_block_size. On the
    // very first call (before any Assign has run for this variable_id) impls/cpu/
    // read_value.cpp falls back to a zero-length allocation, giving old_length == 0,
    // exactly the desired "empty history" start state.
    const cldnn::layout raw_k_decl(ov::PartialShape{B, raw_k_shape[1], Di}, cldnn::data_types::f16, cldnn::format::bfyx);
    const auto raw_k_read_id = layer_name + "_raw_k_read";
    p.add_primitive(*op, cldnn::read_value(raw_k_read_id, {}, raw_k_variable->get_info().variable_id, {raw_k_decl}));

    const cldnn::layout summary_decl(ov::PartialShape{B, summary_shape[1], Di}, cldnn::data_types::f16, cldnn::format::bfyx);
    const auto summary_read_id = layer_name + "_summary_read";
    p.add_primitive(*op, cldnn::read_value(summary_read_id, {}, summary_variable->get_info().variable_id, {summary_decl}));

    std::vector<cldnn::input_info> qsa_inputs(QSAIn::COUNT);
    qsa_inputs[QSAIn::MIXED] = make_zero_constant(p, op, layer_name, "_dummy_mixed", ov::PartialShape{1, std::max<int64_t>(D, 1)}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::QUERY] =
        make_zero_constant(p, op, layer_name, "_dummy_query", ov::PartialShape{1, std::max<int64_t>(Hkv * Dh, 1)}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::KEY] = make_zero_constant(p, op, layer_name, "_dummy_key", ov::PartialShape{1, Hkv * Dh}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::VALUE] = make_zero_constant(p, op, layer_name, "_dummy_value", ov::PartialShape{1, Hkv * Dv}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::PAST_LENS] = make_zero_constant(p, op, layer_name, "_dummy_past_lens", ov::PartialShape{1}, cldnn::data_types::i32);
    qsa_inputs[QSAIn::SUBSEQUENCE_BEGINS] = make_zero_constant(p, op, layer_name, "_dummy_subsequence_begins", ov::PartialShape{2}, cldnn::data_types::i32);
    qsa_inputs[QSAIn::BLOCK_INDICES] = make_iota_constant(p, op, layer_name, "_dummy_block_indices", 1);
    qsa_inputs[QSAIn::BLOCK_INDICES_BEGINS] = make_iota_constant(p, op, layer_name, "_dummy_block_indices_begins", 2);
    qsa_inputs[QSAIn::KEY_CACHE] = make_zero_constant(p, op, layer_name, "_dummy_key_cache", ov::PartialShape{1, Hkv, 1, Dh}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::VALUE_CACHE] = make_zero_constant(p, op, layer_name, "_dummy_value_cache", ov::PartialShape{1, Hkv, 1, Dv}, cldnn::data_types::f16);
    // Fed directly as dense [B, old_length, Di] tensors -- dense_mode's qsa_inst::
    // calc_output_layouts expects rank-3 inputs with no degenerate paged head axis.
    qsa_inputs[QSAIn::INDEXER_RAW_K_CACHE] = cldnn::input_info(raw_k_read_id);
    qsa_inputs[QSAIn::INDEXER_SUMMARY_CACHE] = cldnn::input_info(summary_read_id);
    qsa_inputs[QSAIn::INDEXER_QK_WEIGHT] =
        make_zero_constant(p, op, layer_name, "_dummy_qk_weight", ov::PartialShape{std::max<int64_t>((Hidx + 1) * Di, 1), std::max<int64_t>(D, 1)},
                           cldnn::data_types::f16);
    qsa_inputs[QSAIn::INDEXER_Q_NORM_WEIGHT] =
        make_zero_constant(p, op, layer_name, "_dummy_q_norm_weight", ov::PartialShape{std::max<int64_t>(Di, 1)}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::INDEXER_K_NORM_WEIGHT] =
        make_zero_constant(p, op, layer_name, "_dummy_k_norm_weight", ov::PartialShape{std::max<int64_t>(Di, 1)}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::ROTARY_COS_SIN] = make_zero_constant(p, op, layer_name, "_dummy_rotary", ov::PartialShape{1, 2}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::POSITION_IDS] = make_zero_constant(p, op, layer_name, "_dummy_position_ids", ov::PartialShape{1}, cldnn::data_types::i32);

    cldnn::qsa prim(layer_name, qsa_inputs, config, /*skip_kv_write=*/false, /*prepare_only=*/false, /*dense_mode=*/true, /*state_only=*/true);
    prim.num_outputs = 2;
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);

    // The add_primitive() calls above each overwrote primitive_ids[layer_name] (every
    // add_primitive(op, ...) call remaps op's canonical id to whatever was just added --
    // see ProgramBuilder::add_primitive). Restore it to the qsa primitive itself (id ==
    // layer_name, see the ctor above): it is the one with QSAIndexerCache's actual two
    // declared outputs (raw_k at port 0, summary at port 1), which is what any downstream
    // consumer (e.g. QSASDPA) must resolve to via GetInputInfo().
    p.primitive_ids[layer_name] = layer_name;
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, QSAIndexerCache);

}  // namespace ov::intel_gpu
