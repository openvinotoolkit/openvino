// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"

namespace ov {
namespace intel_cpu {

///
/// \brief jit_container_emitter designed provide common interface for register mapping
/// (abstract to physical) and nested code access.
///
class jit_container_emitter {
public:
    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;

protected:
    // maps gpr and vec abstract registers to physical ones.
    void map_abstract_registers(mapping_info& gpr_map_pool, mapping_info& vec_map_pool, snippets::lowered::LinearIR::container& expressions) const;
};

}   // namespace intel_cpu
}   // namespace ov
