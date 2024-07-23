// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/dimension.hpp"

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

enum class BRGEMM_TYPE {
    STAND_ALONE,            // No extra requirements, used for f32|f32
    WITH_AMX,               // i8|i8 or bf16|bf16 on AMX system - needs BrgemmCopyB and scratchpad
    WITH_COMPENSATIONS,     // i8|i8 (non-AMX system) - needs BrgemmCopyB for data repacking and compensations
    REPACKING_ONLY          // u8|i8 or bf16|bf16 (non-AMX system) - needs BrgemmCopyB on second input for data repacking
};

dnnl::impl::cpu::x64::cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx);

BRGEMM_TYPE get_brgemm_type(const element::Type& element_type_a, const Dimension& K_dim, const Dimension& N_dim);

inline bool stand_alone(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::STAND_ALONE; }

inline bool with_amx(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::WITH_AMX; }

inline bool with_compensations(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::WITH_COMPENSATIONS; }

inline bool repacking_only(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::REPACKING_ONLY; }

inline bool with_repacking(BRGEMM_TYPE type) { return type != BRGEMM_TYPE::STAND_ALONE; }

inline bool with_scratchpad(BRGEMM_TYPE type) { return with_compensations(type) || with_amx(type); }

}   // namespace brgemm_utils
}   // namespace intel_cpu
template <>
class AttributeAdapter<intel_cpu::brgemm_utils::BRGEMM_TYPE> :
        public EnumAttributeAdapterBase<intel_cpu::brgemm_utils::BRGEMM_TYPE> {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BRGEMM_TYPE& value) :
        EnumAttributeAdapterBase<intel_cpu::brgemm_utils::BRGEMM_TYPE>(value) {
    }
    OPENVINO_RTTI("AttributeAdapter<ov::intel_cpu::jit_brgemm_utils::BRGEMM_TYPE>");
};
}   // namespace ov
