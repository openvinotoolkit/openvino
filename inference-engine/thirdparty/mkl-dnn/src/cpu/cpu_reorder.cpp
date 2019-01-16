/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <assert.h>

#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "type_helpers.hpp"

#ifdef MKLDNN_JIT
#include "jit_uni_reorder.hpp"
#endif
#include "simple_reorder.hpp"
#include "wino_reorder.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using rpd_create_f = mkldnn::impl::engine_t::reorder_primitive_desc_create_f;

namespace {
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;

#define REG_SR(idt, ifmt, odt, ofmt, ...) \
    simple_reorder_t<idt, ifmt, odt, ofmt, __VA_ARGS__>::pd_t::create

#define REG_SR_BIDIR(idt, ifmt, odt, ofmt) \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order::keep), \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order::reverse)

#define REG_SR_DIRECT_COPY(idt, odt) \
    REG_SR(idt, any, odt, any, fmt_order::any, spec::direct_copy), \
    REG_SR(idt, any, odt, any, fmt_order::any, spec::direct_copy_except_dim_0)

static const rpd_create_f cpu_reorder_impl_list[] = {
    /* winograd */
    wino_reorder_t<f32, f32>::pd_t::create,
    wino_reorder_t<f32, s8>::pd_t::create,

#ifdef __INTEL_COMPILER
    /* direct copy for icc, which is faster than jitted code */
    REG_SR_DIRECT_COPY(f32, f32),
    REG_SR_DIRECT_COPY(f32, s32),
    REG_SR_DIRECT_COPY(f32, s8),
//    REG_SR_DIRECT_COPY(f32, u8), FIXME: Disabled due to accuracy failure on int8 network
    REG_SR_DIRECT_COPY(s32, f32),
    REG_SR_DIRECT_COPY(s32, s32),
    REG_SR_DIRECT_COPY(s32, s8),
    REG_SR_DIRECT_COPY(s32, u8),
    REG_SR_DIRECT_COPY(s8, f32),
    REG_SR_DIRECT_COPY(s8, s32),
    REG_SR_DIRECT_COPY(s8, s8),
    REG_SR_DIRECT_COPY(s8, u8),
    REG_SR_DIRECT_COPY(u8, f32),
    REG_SR_DIRECT_COPY(u8, s32),
    REG_SR_DIRECT_COPY(u8, s8),
    REG_SR_DIRECT_COPY(u8, u8),
#endif

#ifdef MKLDNN_JIT
    /* jit */
    jit_uni_reorder_create,
#endif

    /* fp32: flat <-> blocked with tail */
    REG_SR_BIDIR(f32, any, f32, nChw8c),
    REG_SR_BIDIR(f32, any, f32, nChw16c),
    REG_SR_BIDIR(f32, any, f32, nCdhw16c),
    REG_SR_BIDIR(f32, nChw8c, f32, nChw16c),

    REG_SR_BIDIR(f32, any, f32, Oihw16o),
    REG_SR_BIDIR(f32, any, f32, Ohwi16o),
    REG_SR_BIDIR(f32, any, f32, Oidhw16o),
    REG_SR_BIDIR(f32, any, f32, Odhwi16o),
    REG_SR_BIDIR(f32, any, f32, OIhw16o16i),
    REG_SR_BIDIR(f32, any, f32, OIhw16i16o),
    REG_SR_BIDIR(f32, any, f32, OIdhw16o16i),
    REG_SR_BIDIR(f32, any, f32, OIdhw16i16o),
    REG_SR_BIDIR(f32, any, f32, IOhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOihw16o),
    REG_SR_BIDIR(f32, any, f32, gOhwi16o),
    REG_SR_BIDIR(f32, any, f32, gOidhw16o),
    REG_SR_BIDIR(f32, any, f32, gOdhwi16o),
    REG_SR_BIDIR(f32, any, f32, gOIhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIhw16i16o),
    REG_SR_BIDIR(f32, any, f32, gOIdhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIdhw16i16o),
    REG_SR_BIDIR(f32, any, f32, gIOhw16o16i),

    /* int: flat <-> blocked with tail */
    REG_SR_BIDIR(f32, nhwc, s32, nChw16c),
    REG_SR_BIDIR(f32, nhwc, s8, nChw16c),
    REG_SR_BIDIR(f32, nhwc, u8, nChw16c),
    REG_SR_BIDIR(s32, nhwc, f32, nChw16c),
    REG_SR_BIDIR(s32, nhwc, s32, nChw16c),
    REG_SR_BIDIR(s32, nhwc, s8, nChw16c),
    REG_SR_BIDIR(s32, nhwc, u8, nChw16c),
    REG_SR_BIDIR(s8, nhwc, f32, nChw16c),
    REG_SR_BIDIR(s8, nhwc, s32, nChw16c),
    REG_SR_BIDIR(s8, nhwc, s8, nChw16c),
    REG_SR_BIDIR(s8, nhwc, u8, nChw16c),
    REG_SR_BIDIR(u8, nhwc, f32, nChw16c),
    REG_SR_BIDIR(u8, nhwc, s32, nChw16c),
    REG_SR_BIDIR(u8, nhwc, s8, nChw16c),
    REG_SR_BIDIR(u8, nhwc, u8, nChw16c),

    REG_SR_BIDIR(f32, oihw, f32, OIhw4i16o4i),
    REG_SR_BIDIR(f32, oihw, s8, OIhw4i16o4i),
    REG_SR_BIDIR(s8, oihw, f32, OIhw4i16o4i),
    REG_SR_BIDIR(s8, oihw, s8, OIhw4i16o4i),
    REG_SR_BIDIR(f32, goihw, s8, gOIhw4i16o4i),
    REG_SR_BIDIR(s8, goihw, f32, gOIhw4i16o4i),
    REG_SR_BIDIR(f32, goihw, f32, gOIhw4i16o4i),
    REG_SR_BIDIR(s8, goihw, s8, gOIhw4i16o4i),

    /* s16 <-> s16 */
    REG_SR_DIRECT_COPY(s16, s16),
    REG_SR_BIDIR(s16, oihw, s16, OIhw8i16o2i),
    REG_SR_BIDIR(s16, goihw, s16, gOIhw8i16o2i),
    REG_SR_BIDIR(s16, OIhw8i16o2i, s16, OIhw8o16i2o),
    REG_SR_BIDIR(s16, gOIhw8i16o2i, s16, gOIhw8o16i2o),

    /* WA to prevent fallback on reference implementations */
    REG_SR_DIRECT_COPY(u8, f32),
    REG_SR_BIDIR(u8, nchw, f32, nChw8c),
    REG_SR_BIDIR(u8, nchw, f32, nChw16c),

    /* reference: the last line of defence */
    REG_SR(f32, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(f32, any, s32, any, fmt_order::any, spec::reference),
    REG_SR(f32, any, s16, any, fmt_order::any, spec::reference),
    REG_SR(f32, any, s8, any, fmt_order::any, spec::reference),
    REG_SR(f32, any, u8, any, fmt_order::any, spec::reference),

    REG_SR(s32, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(s32, any, s32, any, fmt_order::any, spec::reference),
    REG_SR(s32, any, s16, any, fmt_order::any, spec::reference),
    REG_SR(s32, any, s8, any, fmt_order::any, spec::reference),
    REG_SR(s32, any, u8, any, fmt_order::any, spec::reference),

    REG_SR(s16, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(s16, any, s32, any, fmt_order::any, spec::reference),
    REG_SR(s16, any, s16, any, fmt_order::any, spec::reference),

    REG_SR(s8, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(s8, any, s32, any, fmt_order::any, spec::reference),
    REG_SR(s8, any, s8, any, fmt_order::any, spec::reference),
    REG_SR(s8, any, u8, any, fmt_order::any, spec::reference),

    REG_SR(u8, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(u8, any, s32, any, fmt_order::any, spec::reference),
    REG_SR(u8, any, u8, any, fmt_order::any, spec::reference),
    REG_SR(u8, any, s8, any, fmt_order::any, spec::reference),

    /* eol */
    nullptr,
};
}

const rpd_create_f *cpu_engine_t::get_reorder_implementation_list() const {
    return cpu_reorder_impl_list;
}

}
}
}
