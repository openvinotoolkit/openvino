// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fa_utils.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <numeric>
#include <string>

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {

namespace intel_cpu::fa_utils {

FAConfig::FAConfig(const ov::element::Type& src_dt,
                   const ov::element::Type& wei_dt,
                   const ov::element::Type& orig_wei_dt,
                   bool transposed_b)
    : FAConfig(get_prim_isa(src_dt, wei_dt), src_dt, wei_dt, orig_wei_dt, transposed_b) {}

FAConfig::FAConfig(const dnnl::impl::cpu::x64::cpu_isa_t& isa,
                  const ov::element::Type& src_dt,
                  const ov::element::Type& wei_dt,
                  const ov::element::Type& orig_wei_dt,
                  bool transposed_b)
    : m_isa(isa),
      m_src_dt(src_dt),
      m_wei_dt(wei_dt),
      m_orig_wei_dt(orig_wei_dt),
      m_transposed_b(transposed_b) {
    m_q_len_blk = 32;
    m_kv_len_blk = is_superset(m_isa, avx512_core) ? 64 : 24;  // should be the same as brgemm_config, as repack use brgemm_config
}

dnnl::impl::cpu::x64::cpu_isa_t FAConfig::get_prim_isa(const ov::element::Type& src_dt,
                                                       const ov::element::Type& wei_dt) {
#define RETURN_IF_SUPPORTED(x) \
    if (mayiuse(x)) {          \
        return x;              \
    }

    // only f32 precision is supported for now
    OPENVINO_ASSERT(src_dt == ov::element::f32 && wei_dt == ov::element::f32, "Only f32 supported in fa");
    RETURN_IF_SUPPORTED(cpu::x64::avx512_core)
    RETURN_IF_SUPPORTED(cpu::x64::avx2)
    return isa_undef;
#undef RETURN_IF_SUPPORTED
}

}  // namespace intel_cpu::brgemm_utils

}  // namespace ov
