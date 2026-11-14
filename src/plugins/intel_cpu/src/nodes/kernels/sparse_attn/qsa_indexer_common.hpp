// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "onednn/iml_type_mapper.h"
#include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu::node {

// The implementation type of a QSA node that does not score on brgemm: the instruction set its cross-compiled kernel is
// dispatched to on this CPU.
inline impl_desc_type qsa_kernel_impl_type() {
    if (ov::with_cpu_x86_avx512f()) {
        return impl_desc_type::avx512;
    }
    return ov::with_cpu_x86_avx2() ? impl_desc_type::avx2 : impl_desc_type::ref;
}

}  // namespace ov::intel_cpu::node
