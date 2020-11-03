// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn.hpp"
#include "cpu_isa_traits.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace x64 {
    constexpr cpu_isa_t sse42 = cpu_isa_t::sse41; // TODO: clarify is it valid replacement?
}  // namespace x64

    constexpr x64::cpu_isa_t sse42 = x64::sse42;
    constexpr x64::cpu_isa_t avx2 = x64::avx2;
    constexpr x64::cpu_isa_t avx512_common = x64::avx512_common;

}  // namespace cpu
}  // namespace impl
}  // namespace mkldnn
