// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "impls/ocl/kernel_selector_helper.h"
#include "experimental/primitives/draft_block_mask.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

struct DraftBlockMaskImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::draft_block_mask")
    explicit DraftBlockMaskImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
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

        // The ported CM kernels (draft_block_mask_pool/score_softmax/topk_mask) hard-code `half`
        // throughout (no jitted precision templating like the OCL kernels), so only fp16 Q/K is
        // supported for now.
        const auto& q_layout = node.get_input_layout(cldnn::DraftBlockMaskInputIdx::Q_WINDOWS);
        const auto& k_layout = node.get_input_layout(cldnn::DraftBlockMaskInputIdx::K_WINDOWS);
        const auto& mask_layout = node.get_input_layout(cldnn::DraftBlockMaskInputIdx::LOCAL_MASK);

        if (q_layout.data_type != cldnn::data_types::f16 || k_layout.data_type != cldnn::data_types::f16) {
            return false;
        }
        if (mask_layout.data_type != cldnn::data_types::u8 && mask_layout.data_type != cldnn::data_types::i8) {
            return false;
        }

        // 5-input variant (runtime seqlen/topk -- see ov::op::experimental::DraftBlockMask's 5-input
        // constructor): both must be i32/i64, matching the core op's own validation.
        if (node.get_dependencies().size() == 5) {
            const auto& seqlen_layout = node.get_input_layout(cldnn::DraftBlockMaskInputIdx::SEQLEN);
            const auto& topk_layout = node.get_input_layout(cldnn::DraftBlockMaskInputIdx::TOPK);
            const auto is_int = [](cldnn::data_types dt) {
                return dt == cldnn::data_types::i32 || dt == cldnn::data_types::i64;
            };
            if (!is_int(seqlen_layout.data_type) || !is_int(topk_layout.data_type)) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
