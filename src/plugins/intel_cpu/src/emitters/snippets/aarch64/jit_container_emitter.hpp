// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"

#include "snippets/lowered/linear_ir.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

///
/// \brief jit_container_emitter designed to wrap Emitters that contain other Emitters (for example, jit_kernel_emitter)
///  This is needed to provide common interface for register mapping
/// (abstract to physical) and nested code access.
///
class jit_container_emitter: public jit_emitter {
public:
    jit_container_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa);

    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;

protected:
    // maps gpr and vec abstract registers to physical ones.
    void map_abstract_registers(mapping_info& gpr_map_pool, mapping_info& vec_map_pool, snippets::lowered::LinearIR::container& expressions) const;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
