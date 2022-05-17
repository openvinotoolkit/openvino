/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t comp_s8_s8_impl_list_map {
    // s8 -> s8
    {{s8, s8, 2}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))
        REG_REORDER_P(REG_SR(s8, oi, s8, OI4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, io, s8, OI4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oi, s8, OI4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, io, s8, OI4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oi, s8, OI4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, io, s8, OI4i64o4i, fmt_order_keep, spec_conv_req_comp))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, ab, s8, BA16a16b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ab, s8, BA16a32b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ab, s8, BA16a48b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ab, s8, BA16a64b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ba, s8, BA16a16b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ba, s8, BA16a32b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ba, s8, BA16a48b4a, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ba, s8, BA16a64b4a, fmt_order_keep, spec_conv_req_comp))
#endif
        nullptr,
    }},
    // s8 -> s8
    {{s8, s8, 3}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))
        REG_REORDER_P(REG_SR(s8, any, s8, wio, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iwo, s8, Owi16o, fmt_order_keep, spec_conv_req_comp))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, oiw, s8, Owi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, Owi16o, fmt_order_keep, spec_conv_req_comp))
#endif
        REG_REORDER_P(REG_SR(s8, iwo, s8, OwI16o4i, fmt_order_keep, spec_conv_req_comp))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, oiw, s8, OwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OwI16o4i, fmt_order_keep, spec_conv_req_comp))
#endif
        REG_REORDER_P(REG_SR(s8, iwo, s8, OIw16i16o4i, fmt_order_keep, spec_conv_req_comp))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, oiw, s8, OIw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wio, s8, OIw16i16o4i, fmt_order_keep, spec_conv_req_comp))
#endif
        nullptr,
    }},
    {{s8, s8, 4}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, any, s8, hwio, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, any, s8, wigo, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOIw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOIw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOIw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOIw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOIw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOIw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw16g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, Goiw16g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw8g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, Goiw8g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw4g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, Goiw4g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOwi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOwi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goiw, s8, gOIw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, wigo, s8, gOIw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, Owhi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, Owhi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, Owhi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, ihwo, s8, OIhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oihw, s8, OIhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwio, s8, OIhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
#endif
        REG_REORDER_P(REG_SR(s8, iohw, s8, OIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iohw, s8, OIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iohw, s8, OIhw4o4i, fmt_order_keep, spec_conv_req_comp))

        nullptr,
    }},
    {{s8, s8, 5}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, any, s8, hwigo, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, any, s8, dhwio, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw4i32o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw4i64o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw16g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, Goihw16g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw8g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, Goihw8g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw4g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, Goihw4g, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOwhi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOwhi16o, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goihw, s8, gOIhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, hwigo, s8, gOIhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OdhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OdhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OdhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, idhwo, s8, OIdhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, oidhw, s8, OIdhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, dhwio, s8, OIdhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
#endif
        REG_REORDER_P(REG_SR(s8, giohw, s8, gOIhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, giohw, s8, gOIhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, giohw, s8, gOIhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iodhw, s8, OIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iodhw, s8, OIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, iodhw, s8, OIdhw4o4i, fmt_order_keep, spec_conv_req_comp))

        nullptr,
    }},
    {{s8, s8, 6}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t)))
#ifdef ENABLE_UNUSED_PRIM
        REG_REORDER_P(REG_SR(s8, any, s8, dhwigo, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goidhw, s8, gOIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goidhw, s8, gOIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goidhw, s8, gOIdhw4o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goidhw, s8, gOdhwI16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, goidhw, s8, gOIdhw16i16o4i, fmt_order_keep, spec_conv_req_comp))
#endif
        REG_REORDER_P(REG_SR(s8, giodhw, s8, gOIdhw4i16o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, giodhw, s8, gOIdhw2i8o4i, fmt_order_keep, spec_conv_req_comp))
        REG_REORDER_P(REG_SR(s8, giodhw, s8, gOIdhw4o4i, fmt_order_keep, spec_conv_req_comp))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
