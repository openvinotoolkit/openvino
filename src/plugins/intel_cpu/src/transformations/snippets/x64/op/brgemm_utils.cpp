// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx) {
    auto isa = isa_undef;
#define SUPPORT(X, Y) if (mayiuse(X)) { isa = X; } else { Y }
#define SUPPORT_ONE(X, MESSAGE) SUPPORT(X, OV_CPU_JIT_EMITTER_THROW(MESSAGE);)
#define SUPPORT_TWO(X, Y, MESSAGE) SUPPORT(X, SUPPORT_ONE(Y, MESSAGE))
#define SUPPORT_THREE(X, Y, Z, MESSAGE) SUPPORT(X, SUPPORT_TWO(Y, Z, MESSAGE))

    // Note: AMX might be not used even if it's supported by the hardware, check the BrgemmToBrgemmCPU pass for details
    if (is_with_amx) {
        SUPPORT_ONE(avx512_core_amx, "Unsupported hardware configuration: amx is supported only on avx512 platforms")
    } else if (dt_in0 == ov::element::bf16) {
        SUPPORT_ONE(avx512_core_bf16, "Unsupported hardware configuration: bf16 is supported only on avx512 platforms")
    } else if (one_of(dt_in0, ov::element::u8, ov::element::i8)) {
        SUPPORT_THREE(avx512_core_vnni, avx2_vnni_2, avx2_vnni, "Unsupported hardware configuration: int8 is supported only on vnni platforms")
    } else {
        SUPPORT_TWO(avx512_core, cpu::x64::avx2, "Unsupported hardware configuration: brgemm requires at least avx2 isa")
    }
    return isa;
#undef SUPPORT_TWO
#undef SUPPORT_ONE
#undef SUPPORT
}

BRGEMM_TYPE get_brgemm_type(const ov::element::Type& element_type_a, const Dimension& K_dim, bool transpose_b) {
    if (element_type_a == element::f32)
        return transpose_b ? BRGEMM_TYPE::REPACKING_ONLY : BRGEMM_TYPE::STAND_ALONE;

    OPENVINO_ASSERT(element_type_a != element::bf16 || mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16),
                    "BF16 precision is not supported on this hardware");

    const auto brgemmVNNIFactor = 4 / element_type_a.size();
    if (one_of(element_type_a, element::u8, element::i8, element::bf16) &&
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
        K_dim.is_static() && K_dim.get_length() % brgemmVNNIFactor == 0)
        return BRGEMM_TYPE::WITH_AMX;
    // Note: this condition reproduces logic from the OneDNN Brgemm implementation. This is needed to align with the
    // backend requirements. More details in onednn/src/cpu/x64/brgemm/brgemm_utils.cpp
    if (element_type_a == ov::element::i8)
       return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2) ? BRGEMM_TYPE::REPACKING_ONLY : BRGEMM_TYPE::WITH_COMPENSATIONS;

    if (one_of(element_type_a, element::u8, ov::element::bf16))
        return BRGEMM_TYPE::REPACKING_ONLY;
    OV_CPU_JIT_EMITTER_THROW("Failed to determine brgemm mode");
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

size_t get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen = x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

namespace repacking {
size_t compute_out_leading_dim(const size_t n_block, const ov::element::Type& precision) {
    return std::max(n_block, compute_inner_n_block(precision));
}

size_t compute_inner_n_block(const ov::element::Type& precision) {
    switch (precision) {
        case element::i8: return 64;
        case element::bf16: return 32;
        case element::f32: return 16;
        default: OPENVINO_THROW("BrgemmCopyB doesn't support precision ", precision);
    }
}
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu
template <>
EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>& EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>::get() {
    static auto enum_names =
            EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>("ov::intel_cpu::jit_bgremm_utils::BRGEMM_TYPE",
                                                                {{"stand_alone", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::STAND_ALONE},
                                                                 {"with_amx", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_AMX},
                                                                 {"with_compensations", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_COMPENSATIONS},
                                                                 {"repacking_only", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::REPACKING_ONLY}});
    return enum_names;
}
}   // namespace ov
