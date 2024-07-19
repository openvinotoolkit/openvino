// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/dimension.hpp"

namespace ov {
namespace intel_cpu {
namespace jit_brgemm_utils {

enum class BRGEMM_TYPE { STAND_ALONE, WITH_AMX, WITH_COMPENSATIONS, WITH_REPACKING };

dnnl::impl::cpu::x64::cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx);

BRGEMM_TYPE get_brgemm_type(const element::Type& element_type_a, const Dimension& K_dim, const Dimension& N_dim);

inline bool stand_alone(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::STAND_ALONE; }

inline bool with_amx(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::WITH_AMX; }

inline bool with_compensations(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::WITH_COMPENSATIONS; }

inline bool with_repacking(BRGEMM_TYPE type) { return type == BRGEMM_TYPE::WITH_REPACKING; }

}   // namespace jit_brgemm_utils
}   // namespace intel_cpu
}   // namespace ov
