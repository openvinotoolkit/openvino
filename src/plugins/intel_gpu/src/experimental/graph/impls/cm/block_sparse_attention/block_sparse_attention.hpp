// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "experimental/primitives/block_sparse_attention.hpp"
#include "impls/ocl/kernel_selector_helper.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

struct BlockSparseAttentionImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::block_sparse_attention")
    explicit BlockSparseAttentionImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override {
        if (node.has_fused_primitives()) {
            return false;
        }

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();
        // CM optimized for systolic-array architectures -- same gate as every other CM manager
        // (see e.g. impls/cm/paged_attention.hpp, impls/cm/vl_sdpa_opt.hpp). Without it, this impl
        // could be selected on hardware without the systolic array / CM JIT, or when the user
        // disabled CM.
        if (!cldnn::check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
            return false;
        }

        // The ported CM kernel (block_sparse_attention_prefill) hard-codes `half` throughout (no
        // jitted precision templating like the OCL kernels), so only fp16 Q/K/V is supported for
        // now.
        const auto& q_layout = node.get_input_layout(cldnn::BlockSparseAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(cldnn::BlockSparseAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(cldnn::BlockSparseAttentionInputIdx::VALUE);
        const auto& mask_layout = node.get_input_layout(cldnn::BlockSparseAttentionInputIdx::BLOCK_MASK);

        if (q_layout.data_type != cldnn::data_types::f16 || k_layout.data_type != cldnn::data_types::f16 || v_layout.data_type != cldnn::data_types::f16) {
            return false;
        }
        if (mask_layout.data_type != cldnn::data_types::u8) {
            return false;
        }

        // The ported kernel's prefill path only handles an fp16 scale (the same hard-coded `half`
        // convention as Q/K/V above); reject any other scale input when one is present (5-input
        // form).
        if (node.get_dependencies().size() == 5) {
            const auto& scale_layout = node.get_input_layout(cldnn::BlockSparseAttentionInputIdx::SCALE);
            if (scale_layout.data_type != cldnn::data_types::f16) {
                return false;
            }
        }

        // block_sparse_attention_prefill.cm's SPARSE_BLOCK_SIZE is a single scalar shared by both
        // the Q- and KV-block granularity (mirroring pageatten/cm_pa_xe2.hpp's own convention), and
        // must be 128 or 256 (see cm_sdpa_common.hpp's sdpa_kernel_lsc_prefetch static_assert) and a
        // multiple of q_step(16) so a work-group's Q tile never straddles a sparse-block boundary.
        const auto desc = node.as<cldnn::block_sparse_attention>().get_primitive();
        if (desc->block_size_q != desc->block_size_kv) {
            return false;
        }
        if (desc->block_size_q != 128 && desc->block_size_q != 256) {
            return false;
        }

        // v1 limitation: the ported kernel's sparse_block_mask indexing assumes a dense
        // [B,Hq,num_q_blocks,num_k_blocks] mask (no Bm==1/Hm==1|Hkv broadcast support); reject
        // broadcast shapes up front when they're statically knowable (checked again, fatally, at
        // dispatch time for the dynamic case -- see block_sparse_attention.cpp).
        const auto& mask_shape = mask_layout.get_partial_shape();
        const auto& q_shape = q_layout.get_partial_shape();
        if (mask_shape.rank().is_static() && q_shape.rank().is_static()) {
            if (mask_shape[0].is_static() && q_shape[0].is_static() && mask_shape[0].get_length() != q_shape[0].get_length()) {
                return false;
            }
            if (mask_shape[1].is_static() && q_shape[2].is_static() && mask_shape[1].get_length() != q_shape[2].get_length()) {
                return false;
            }
        }

        // v1 limitation: the ported kernels use a single CMFLA_HEAD_SIZE (= E, query/key's head
        // size) throughout for output and scratch addressing -- a distinct V head size (Ev != E,
        // which the core op/spec otherwise allow) mis-addresses and can write out of bounds. Reject
        // it up front when statically knowable.
        const auto& v_shape = v_layout.get_partial_shape();
        if (q_shape.rank().is_static() && v_shape.rank().is_static() && q_shape[3].is_static() && v_shape[3].is_static() &&
            q_shape[3].get_length() != v_shape[3].get_length()) {
            return false;
        }

        // v1 limitation: the prefill kernel's DPAS REG_K == 16 block tiling and LSC 2D-Block
        // descriptor width/pitch alignment require head_size (E) to be a multiple of 16 -- a
        // head_size that isn't a multiple of REG_K would issue mis-aligned cm_svm_block_read /
        // cm_svm_block_write that overflow past the query/output buffer's head-size stride. This
        // constraint comes from the (now sole) prefill path, which is the kernel every Lq
        // -- including single-token Lq==1 -- runs on, so it is enforced here at validate_impl time
        // (checked again, fatally, at dispatch time in resolve_geometry()).
        if (q_shape.rank().is_static() && q_shape[3].is_static() && q_shape[3].get_length() % 16 != 0) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
