// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/type.hpp"
#include "primitive_ocl_base.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_utils.hpp"
#include "utils/kernel_generator.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct SDPABase : public KernelGenerator {
    SDPABase(std::string_view name, std::string_view suffix, bool indirect) : KernelGenerator(name, suffix), m_indirect(indirect) {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;

    [[nodiscard]] std::pair<int64_t, int64_t> get_gqa_params(const kernel_impl_params& params) const;
    bool m_indirect;
};

class SDPAImplBase : public PrimitiveImplOCL {
public:
    static constexpr const size_t INDIRECT_STAGE = 10;
    static constexpr const size_t REGULAR_STAGE = 20;

    explicit SDPAImplBase(const std::string& name) : PrimitiveImplOCL(name) {}
    explicit SDPAImplBase(const ov::DiscreteTypeInfo& info) : PrimitiveImplOCL(std::string(info.name)) {}

    static size_t get_beam_table_id(const std::shared_ptr<const scaled_dot_product_attention>& primitive) {
        return primitive->input_size() - 1;
    }

    static bool need_indirect_load(const scaled_dot_product_attention_inst& instance) {
        auto desc = instance.get_typed_desc<scaled_dot_product_attention>();

        if (!instance.has_indirect_inputs()) {
            return false;
        }

        const auto& params = *instance.get_impl_params();
        const auto indirect_axis = desc->indirect_axis;
        if (params.input_layouts[get_beam_table_id(desc)].get_partial_shape()[indirect_axis].get_length() == 1) {
            return false;
        }

        const auto& deps = instance.dependencies();

        const auto indirect_dep_idx = 1;
        const auto& indirect_dep = deps[indirect_dep_idx].first;
        if (dynamic_cast<const kv_cache_inst*>(indirect_dep) == nullptr) {
            return true;
        }

        auto state_layout = indirect_dep->get_impl_params()->get_input_layout(0);
        bool is_prefill = state_layout.count() == 0;
        return !is_prefill;
    }
};

}  // namespace ov::intel_gpu::ocl
