// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "onednn/dnnl.h"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass {

ExternalRepackingAdjusterBase::ExternalRepackingAdjusterBase(const CPURuntimeConfigurator* configurator,
                                                             std::string itt_name)
    : ov::snippets::lowered::pass::RuntimeOptimizer(configurator),
      m_itt_name(std::move(itt_name)) {}

void ExternalRepackingAdjusterBase::register_repacked_input(size_t idx, bool needs_runtime_repacking) {
    m_repacked_inputs.emplace(idx, needs_runtime_repacking);
}

bool ExternalRepackingAdjusterBase::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, m_itt_name.c_str())
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());

    size_t data_size = 0;
    bool has_runtime_repacking = false;
    for (const auto& [idx, needs_runtime_repacking] : m_repacked_inputs) {
        if (!needs_runtime_repacking) {
            continue;
        }
        has_runtime_repacking = true;
        data_size += update_runtime_repacking_data_size(linear_ir, *cpu_config, idx);
    }

    if (!has_runtime_repacking) {
        cpu_config->repacking_impl_type = CPURuntimeConfig::RepackingImplType::NONE;
    } else {
        const auto cache_size = dnnl::utils::get_cache_size(1, true) + dnnl::utils::get_cache_size(2, true);
        const auto fit_into_cache = data_size < cache_size;
        cpu_config->repacking_impl_type = fit_into_cache ? CPURuntimeConfig::RepackingImplType::IN_PARALLEL
                                                         : CPURuntimeConfig::RepackingImplType::SEPARATE;
    }

    const auto is_impl_parallel = cpu_config->repacking_impl_type == CPURuntimeConfig::RepackingImplType::IN_PARALLEL;
    for (const auto& [idx, needs_runtime_repacking] : m_repacked_inputs) {
        if (needs_runtime_repacking) {
            update_runtime_repacking_input(linear_ir, *cpu_config, idx, is_impl_parallel);
        } else {
            update_compile_time_repacked_input(linear_ir, *cpu_config, idx);
        }
    }

    return true;
}

}  // namespace ov::intel_cpu::pass
