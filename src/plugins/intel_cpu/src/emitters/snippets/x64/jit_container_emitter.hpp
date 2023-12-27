// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include "snippets/lowered/linear_ir.hpp"

namespace ov {
namespace intel_cpu {

///
/// \brief jit_container_emitter designed to wrap Emitters that contain other Emitters (for example, jit_kernel_emitter)
///  This is needed to provide common interface for register mapping
/// (abstract to physical) and nested code access.
///
class jit_container_emitter: public jit_emitter {
public:
    jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa);

    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;

protected:
    // maps gpr and vec abstract registers to physical ones. Physical reg indexes are taken from the provided pools
    // (the first 2 args). All the used gpr and vec registers are also stored in the provided sets (the second 2 args).
    void map_abstract_registers(mapping_info& gpr_map_pool, mapping_info& vec_map_pool, snippets::lowered::LinearIR::container& expressions) const;

    snippets::lowered::LinearIR body;
};

}   // namespace intel_cpu
}   // namespace ov
