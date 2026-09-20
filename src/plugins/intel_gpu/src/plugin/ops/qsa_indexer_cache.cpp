// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa_indexer_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/qsa.hpp"
#include "intel_gpu/primitives/read_value.hpp"

// NOTE: QSAIndexerCache has no ov::op::internal counterpart (like ov::intel_gpu::op::
// KVCache, and unlike QSA/QSASDPA/GatedResidual): it is only ever produced internally by a
// future graph-fusion pass, never constructed directly by a model author, so it does not
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

using QSAIndexerCacheIn = ov::intel_gpu::op::QSAIndexerCache::InputIdx;
using QSAIn = ov::intel_gpu::op::QSA::InputIdx;

// See ops/qsa_sdpa.cpp for the identical helpers (kept as separate, self-contained
// copies -- each ops/*.cpp file in this directory is independent, no shared internal
// lowering-helpers header exists yet).

cldnn::input_info make_zero_constant(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& layer_name, const std::string& suffix,
                                     const ov::PartialShape& shape, cldnn::data_types dtype) {
    OPENVINO_ASSERT(shape.is_static(), "[GPU] QSAIndexerCache '", layer_name, "': dummy constant '", suffix, "' requires a static shape");
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

void CreateQSAIndexerCacheOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::QSAIndexerCache>& op) {
    validate_inputs_count(op, {QSAIndexerCacheIn::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& config = op->get_config();

    auto raw_k_variable = op->get_variable();
    auto summary_variable = op->get_summary_variable();
    OPENVINO_ASSERT(raw_k_variable && summary_variable, "[GPU] QSAIndexerCache '", layer_name, "' requires both a raw-K and a summary Variable");

    const int64_t Di = static_cast<int64_t>(config.indexer_head_dim);
    const int64_t r = static_cast<int64_t>(config.compress_ratio);
    OPENVINO_ASSERT(r > 0, "[GPU] QSAIndexerCache '", layer_name, "': compress_ratio must be > 0");

    // QSAIndexerCache dynamic growth: raw-K/summary are genuinely
    // dynamic-length runtime state -- capacity grows call-over-call (see qsa_inst::
    // calc_output_layouts' dense_mode branch and impls/cm/qsa/qsa.cpp's execute()
    // copy_from()-based relocation), NOT pre-reserved to a fixed pa_block_size Variable
    // like the previous (replaced) degenerate-paging lowering. Only the batch and
    // indexer-head-dim axes need to be statically known; the sequence/summary-count axis
    // is dynamic.
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

    const auto& past_lens_ps = op->get_input_partial_shape(QSAIndexerCacheIn::PAST_LENS);
    OPENVINO_ASSERT(past_lens_ps.rank().is_static() && past_lens_ps[0].is_static() && past_lens_ps[0].get_length() == B,
                    "[GPU] QSAIndexerCache '", layer_name, "': PAST_LENS's batch dimension must be static and equal to the Variables' (", B, ")");

    const int64_t Hkv = static_cast<int64_t>(config.kv_heads_num);
    const int64_t Dh = static_cast<int64_t>(config.k_head_size);
    const int64_t Dv = static_cast<int64_t>(config.v_head_size);

    // ReadValue (0 inputs): returns the Variable's CURRENT valid-length content. Unlike
    // the previous fixed-capacity implementation, the declared layout's sequence axis
    // reuses the Variable's OWN declared dimension as-is (static, bounded-dynamic, or
    // fully dynamic -- whatever the caller declared it as; see qsa_indexer_cache.hpp's
    // class comment) rather than a fixed pa_block_size. On the very first call (before
    // any Assign has run for this variable_id) impls/cpu/read_value.cpp falls back to a
    // zero-length allocation, giving old_length == 0, exactly the desired "empty
    // history" start state.
    const cldnn::layout raw_k_decl(ov::PartialShape{B, raw_k_shape[1], Di}, cldnn::data_types::f16, cldnn::format::bfyx);
    const auto raw_k_read_id = layer_name + "_raw_k_read";
    p.add_primitive(*op, cldnn::read_value(raw_k_read_id, {}, raw_k_variable->get_info().variable_id, {raw_k_decl}));

    const cldnn::layout summary_decl(ov::PartialShape{B, summary_shape[1], Di}, cldnn::data_types::f16, cldnn::format::bfyx);
    const auto summary_read_id = layer_name + "_summary_read";
    p.add_primitive(*op, cldnn::read_value(summary_read_id, {}, summary_variable->get_info().variable_id, {summary_decl}));

    std::vector<cldnn::input_info> qsa_inputs(QSAIn::COUNT);
    qsa_inputs[QSAIn::MIXED] = inputs[QSAIndexerCacheIn::MIXED];
    // QSAIn::QUERY is never read by Q1 (prepare_only never dispatches Q2/Q3, the only
    // stages that would); its declared VALUES are irrelevant. Its ROW COUNT, however, IS
    // read by get_internal_buffer_descs() as this call's "tokens" (to size Q1's own
    // PREPARE_MAP), and that row count is dynamic (varies per call, same as MIXED's), so
    // a fixed-size dummy constant is wrong here (unlike KEY/VALUE below, which are never
    // read for ANY purpose, not even shape). Reusing MIXED's own graph output directly
    // guarantees the row count always matches with zero extra graph nodes; the column
    // width mismatch (D vs Hq*Dh) is harmless since nothing reads QUERY's shape[1] or
    // contents in this mode.
    qsa_inputs[QSAIn::QUERY] = inputs[QSAIndexerCacheIn::MIXED];
    qsa_inputs[QSAIn::KEY] = make_zero_constant(p, op, layer_name, "_dummy_key", ov::PartialShape{1, Hkv * Dh}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::VALUE] = make_zero_constant(p, op, layer_name, "_dummy_value", ov::PartialShape{1, Hkv * Dv}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::PAST_LENS] = inputs[QSAIndexerCacheIn::PAST_LENS];
    qsa_inputs[QSAIn::SUBSEQUENCE_BEGINS] = inputs[QSAIndexerCacheIn::SUBSEQUENCE_BEGINS];
    // BLOCK_INDICES/BLOCK_INDICES_BEGINS are structurally dead in dense_mode (the
    // degenerate paged-address helpers are still called with page==0 internally, but
    // dense_mode derives QSA_PA_BLOCK_SIZE from the runtime cache layout rather than
    // consulting these tables at all -- see qsa.cpp's dense_pa_block_size()); a single
    // fixed-size placeholder is enough regardless of the real batch size.
    qsa_inputs[QSAIn::BLOCK_INDICES] = make_iota_constant(p, op, layer_name, "_block_indices", 1);
    qsa_inputs[QSAIn::BLOCK_INDICES_BEGINS] = make_iota_constant(p, op, layer_name, "_block_indices_begins", 2);
    // KEY_CACHE/VALUE_CACHE are dead in prepare_only mode (only Q2/Q3 read them, and
    // neither is ever dispatched here); a minimal static placeholder is enough since
    // dense_mode's growth math never inspects their shape either.
    qsa_inputs[QSAIn::KEY_CACHE] = make_zero_constant(p, op, layer_name, "_dummy_key_cache", ov::PartialShape{1, Hkv, 1, Dh}, cldnn::data_types::f16);
    qsa_inputs[QSAIn::VALUE_CACHE] = make_zero_constant(p, op, layer_name, "_dummy_value_cache", ov::PartialShape{1, Hkv, 1, Dv}, cldnn::data_types::f16);
    // Fed directly as dense [B, old_length, Di] tensors -- dense_mode's qsa_inst::
    // calc_output_layouts expects rank-3 inputs with no degenerate paged head axis, so
    // (unlike the previous implementation) no reshape is inserted here.
    qsa_inputs[QSAIn::INDEXER_RAW_K_CACHE] = cldnn::input_info(raw_k_read_id);
    qsa_inputs[QSAIn::INDEXER_SUMMARY_CACHE] = cldnn::input_info(summary_read_id);
    qsa_inputs[QSAIn::INDEXER_QK_WEIGHT] = inputs[QSAIndexerCacheIn::INDEXER_QK_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_Q_NORM_WEIGHT] = inputs[QSAIndexerCacheIn::INDEXER_Q_NORM_WEIGHT];
    qsa_inputs[QSAIn::INDEXER_K_NORM_WEIGHT] = inputs[QSAIndexerCacheIn::INDEXER_K_NORM_WEIGHT];
    qsa_inputs[QSAIn::ROTARY_COS_SIN] = inputs[QSAIndexerCacheIn::ROTARY_COS_SIN];
    qsa_inputs[QSAIn::POSITION_IDS] = inputs[QSAIndexerCacheIn::POSITION_IDS];

    cldnn::qsa prim(layer_name, qsa_inputs, config, /*skip_kv_write=*/false, /*prepare_only=*/true, /*dense_mode=*/true);
    prim.num_outputs = 2;
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);

    // Write the (now Q1-grown) raw-K/summary buffers back to their Variables. The
    // declared layout's sequence axis stays dynamic here too -- it only communicates the
    // Variable's persistent rank/dtype, not this call's concrete length, which
    // execute()/calc_output_layouts already resolved for qsa's own two outputs.
    //
    // These Assign primitives have no consumers of their own (nothing in the OV graph
    // reads an "assign result" -- their entire purpose is the side effect on Variable
    // state), but they DO have real dependencies (qsa's output0/output1), so cldnn's
    // dangling-node removal (program::remove_if_dangling, which requires BOTH empty
    // users AND empty dependencies to prune) does not remove them.
    p.add_primitive(*op, cldnn::assign(layer_name + "_raw_k_assign", {cldnn::input_info(layer_name, 0)}, raw_k_variable->get_info().variable_id,
                                       raw_k_decl));
    p.add_primitive(*op, cldnn::assign(layer_name + "_summary_assign", {cldnn::input_info(layer_name, 1)}, summary_variable->get_info().variable_id,
                                       summary_decl));

    // The add_primitive() calls above each overwrote primitive_ids[layer_name] (every
    // add_primitive(op, ...) call remaps op's canonical id to whatever was just added --
    // see ProgramBuilder::add_primitive). Restore it to the qsa primitive itself (id ==
    // layer_name, see the ctor above): it is the one with QSAIndexerCache's actual two
    // declared outputs (raw_k_dense at port 0, summary_dense at port 1), which is what any
    // downstream consumer (e.g. QSASDPA) must resolve to via GetInputInfo().
    p.primitive_ids[layer_name] = layer_name;
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, QSAIndexerCache);

}  // namespace ov::intel_gpu
