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

#include "cpu/jit_uni_reorder.hpp"
#include "cpu/simple_reorder.hpp"
#include "cpu/wino_reorder.hpp"
#include "cpu/rnn/rnn_reorders.hpp"

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

    /* rnn reorders */
    rnn_data_reorder_t<f32, u8>::pd_t::create,
    rnn_weights_reorder_t<f32, f32>::pd_t::create,
    rnn_weights_reorder_t<f32, s8>::pd_t::create,

#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && !defined(__clang__))
    /* Direct copy for icc which is faster than jitted code;
     * Direct copy for gcc which might or might not be faster than jitted
     * code, but still worth it because doesn't require jitting, i.e. much
     * faster creation time. This is tentative solution and should be removed
     * later (when we will cache jitted code?...). */
    REG_SR_DIRECT_COPY(f32, f32),
#endif

#ifdef __INTEL_COMPILER
    /* direct copy for icc, which is faster than jitted code */
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

    /* jit */
    jit_uni_reorder_create,

    /* fp32: flat <-> blocked with tail */
    REG_SR_BIDIR(f32, any, f32, nCw4c),

    REG_SR_BIDIR(f32, nchw, bin, nhwc),
    REG_SR_BIDIR(f32, nhwc, bin, nhwc),
    REG_SR_DIRECT_COPY(bin, bin),

    REG_SR_BIDIR(f32, any, f32, nCw8c),
    REG_SR_BIDIR(f32, any, f32, OIw4i4o),
    REG_SR_BIDIR(f32, any, f32, OIw8i8o),
    REG_SR_BIDIR(f32, any, f32, OIw8o8i),
    REG_SR_BIDIR(f32, any, f32, gOIw4i4o),
    REG_SR_BIDIR(f32, any, f32, gOIw8i8o),
    REG_SR_BIDIR(f32, any, f32, gOIw8o8i),

    REG_SR_BIDIR(f32, any, f32, nCw16c),
    REG_SR_BIDIR(f32, any, f32, OIw16o16i),
    REG_SR_BIDIR(f32, any, f32, OIw16i16o),
    REG_SR_BIDIR(f32, any, f32, IOw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIw16i16o),
    REG_SR_BIDIR(f32, any, f32, gIOw16o16i),

    REG_SR_BIDIR(f32, any, f32, nChw4c),
    REG_SR_BIDIR(f32, any, f32, nChw8c),
    REG_SR_BIDIR(f32, any, f32, OIhw4i4o),
    REG_SR_BIDIR(f32, any, f32, Ohwi8o),
    REG_SR_BIDIR(f32, any, f32, OIhw8i8o),
    REG_SR_BIDIR(f32, any, f32, OIhw8o8i),
    REG_SR_BIDIR(f32, any, f32, gOIhw4i4o),
    REG_SR_BIDIR(f32, any, f32, gOIhw4o4i),
    REG_SR_BIDIR(f32, any, f32, gOhwi8o),
    REG_SR_BIDIR(f32, any, f32, gOIhw8i8o),
    REG_SR_BIDIR(f32, any, f32, gOIhw8o8i),

    REG_SR_BIDIR(f32, any, f32, nChw16c),
    REG_SR_BIDIR(f32, any, f32, Oihw4o),
    REG_SR_BIDIR(f32, any, f32, Oihw16o),
    REG_SR_BIDIR(f32, any, f32, Ohwi4o),
    REG_SR_BIDIR(f32, any, f32, Ohwi16o),
    REG_SR_BIDIR(f32, any, f32, OIhw16o16i),
    REG_SR_BIDIR(f32, any, f32, OIhw16i16o),
    REG_SR_BIDIR(f32, any, f32, IOhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOihw4o),
    REG_SR_BIDIR(f32, any, f32, gOihw16o),
    REG_SR_BIDIR(f32, any, f32, gOhwi4o),
    REG_SR_BIDIR(f32, any, f32, gOhwi16o),
    REG_SR_BIDIR(f32, any, f32, gOIhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIhw16i16o),
    REG_SR_BIDIR(f32, any, f32, gIOhw16o16i),

    REG_SR_BIDIR(f32, any, f32, nCdhw4c),
    REG_SR_BIDIR(f32, any, f32, nCdhw8c),
    REG_SR_BIDIR(f32, any, f32, OIdhw4i4o),
    REG_SR_BIDIR(f32, any, f32, Odhwi8o),
    REG_SR_BIDIR(f32, any, f32, OIdhw8i8o),
    REG_SR_BIDIR(f32, any, f32, OIdhw8o8i),
    REG_SR_BIDIR(f32, any, f32, gOIdhw4i4o),
    REG_SR_BIDIR(f32, any, f32, gOdhwi8o),
    REG_SR_BIDIR(f32, any, f32, gOIdhw8i8o),
    REG_SR_BIDIR(f32, any, f32, gOIdhw8o8i),

    REG_SR_BIDIR(f32, any, f32, nCdhw16c),
    REG_SR_BIDIR(f32, any, f32, Oidhw4o),
    REG_SR_BIDIR(f32, any, f32, Oidhw16o),
    REG_SR_BIDIR(f32, any, f32, Odhwi16o),
    REG_SR_BIDIR(f32, any, f32, OIdhw16o16i),
    REG_SR_BIDIR(f32, any, f32, OIdhw16i16o),
    REG_SR_BIDIR(f32, any, f32, gOidhw4o),
    REG_SR_BIDIR(f32, any, f32, gOidhw16o),
    REG_SR_BIDIR(f32, any, f32, gOdhwi16o),
    REG_SR_BIDIR(f32, any, f32, gOIdhw16o16i),
    REG_SR_BIDIR(f32, any, f32, gOIdhw16i16o),

    /* WA to prevent fallback on reference implementations */
    REG_SR_DIRECT_COPY(u8, f32),
    REG_SR_DIRECT_COPY(u8, s8),
    REG_SR_DIRECT_COPY(s8, u8),
    REG_SR_DIRECT_COPY(u8, u8),
    REG_SR_DIRECT_COPY(s8, s8),

 /* fp32: blocked <-> blocked with tail */
    REG_SR_BIDIR(f32, nCw8c, f32, nCw16c),
    REG_SR_BIDIR(f32, nChw4c, f32, nChw16c),
    REG_SR_BIDIR(f32, nChw8c, f32, nChw16c),
    REG_SR_BIDIR(f32, nCdhw8c, f32, nCdhw16c),

    /* int: flat <-> blocked with tail */
    REG_SR(f32, nChw8c, u8, nhwc, fmt_order::keep),
    REG_SR(f32, nChw8c, s8, nhwc, fmt_order::keep),
    REG_SR(u8, nhwc, f32, nChw8c, fmt_order::keep),
    REG_SR(s8, nhwc, f32, nChw8c, fmt_order::keep),
    REG_SR(f32, nhwc, u8, nhwc, fmt_order::keep),
    REG_SR(f32, nhwc, s8, nhwc, fmt_order::keep),
    REG_SR(u8, nhwc, f32, nhwc, fmt_order::keep),
    REG_SR(s8, nhwc, f32, nhwc, fmt_order::keep),
    REG_SR(s8, nhwc, u8, nhwc, fmt_order::keep),
    REG_SR(u8, nhwc, s8, nhwc, fmt_order::keep),
    REG_SR(u8, nhwc, s8, nhwc, fmt_order::keep),
    REG_SR(f32, nchw, u8, nhwc, fmt_order::keep),
    REG_SR(f32, nchw, s8, nhwc, fmt_order::keep),
    REG_SR(u8, nchw, u8, nhwc, fmt_order::keep),
    REG_SR(s8, nchw, s8, nhwc, fmt_order::keep),
    REG_SR(u8, nhwc, f32, nchw, fmt_order::keep),

    REG_SR_BIDIR(f32, any, s32, nChw8c),
    REG_SR_BIDIR(f32, any, s8, nChw8c),
    REG_SR_BIDIR(f32, any, u8, nChw8c),
    REG_SR_BIDIR(s32, any, f32, nChw8c),
    REG_SR_BIDIR(s32, any, s32, nChw8c),
    REG_SR_BIDIR(s32, any, s8, nChw8c),
    REG_SR_BIDIR(s32, any, u8, nChw8c),
    REG_SR_BIDIR(s8, any, f32, nChw8c),
    REG_SR_BIDIR(s8, any, s32, nChw8c),
    REG_SR_BIDIR(s8, any, s8, nChw8c),
    REG_SR_BIDIR(s8, any, u8, nChw8c),
    REG_SR_BIDIR(u8, any, f32, nChw8c),
    REG_SR_BIDIR(u8, any, s32, nChw8c),
    REG_SR_BIDIR(u8, any, s8, nChw8c),
    REG_SR_BIDIR(u8, any, u8, nChw8c),

    REG_SR_BIDIR(f32, any, s32, nChw16c),
    REG_SR_BIDIR(f32, any, s8, nChw16c),
    REG_SR_BIDIR(f32, any, u8, nChw16c),
    REG_SR_BIDIR(s32, any, f32, nChw16c),
    REG_SR_BIDIR(s32, any, s32, nChw16c),
    REG_SR_BIDIR(s32, any, s8, nChw16c),
    REG_SR_BIDIR(s32, any, u8, nChw16c),
    REG_SR_BIDIR(s8, any, f32, nChw16c),
    REG_SR_BIDIR(s8, any, s32, nChw16c),
    REG_SR_BIDIR(s8, any, s8, nChw16c),
    REG_SR_BIDIR(s8, any, u8, nChw16c),
    REG_SR_BIDIR(u8, any, f32, nChw16c),
    REG_SR_BIDIR(u8, any, s32, nChw16c),
    REG_SR_BIDIR(u8, any, s8, nChw16c),
    REG_SR_BIDIR(u8, any, u8, nChw16c),

    REG_SR_BIDIR(f32, any, f32, OIhw4i16o4i),
    REG_SR_BIDIR(f32, any, s8, OIhw4i16o4i),
    REG_SR_BIDIR(s8, any, f32, OIhw4i16o4i),
    REG_SR_BIDIR(s8, any, s8, OIhw4i16o4i),
    REG_SR_BIDIR(f32, any, s8, gOIhw4i16o4i),
    REG_SR_BIDIR(s8, any, f32, gOIhw4i16o4i),
    REG_SR_BIDIR(f32, any, f32, gOIhw4i16o4i),
    REG_SR_BIDIR(s8, any, s8, gOIhw4i16o4i),

    REG_SR(f32, any, f32, OhIw8o4i, fmt_order::keep),
    REG_SR(f32, any, s8, OhIw8o4i, fmt_order::keep),
    REG_SR(s8, any, f32, OhIw8o4i, fmt_order::keep),
    REG_SR(s8, any, s8, OhIw8o4i, fmt_order::keep),
    REG_SR(f32, any, s8, gOhIw8o4i, fmt_order::keep),
    REG_SR(s8, any, f32, gOhIw8o4i, fmt_order::keep),
    REG_SR(f32, any, f32, gOhIw8o4i, fmt_order::keep),
    REG_SR(s8, any, s8, gOhIw8o4i, fmt_order::keep),
    REG_SR(f32, oihw, s8, OhIw8o4i_s8s8, fmt_order::keep),
    REG_SR(s8, oihw, s8, OhIw8o4i_s8s8, fmt_order::keep),
    REG_SR(f32, goihw, s8, gOhIw8o4i_s8s8, fmt_order::keep),
    REG_SR(s8, goihw, s8, gOhIw8o4i_s8s8, fmt_order::keep),

    REG_SR(bin, any, bin, OhIw8o32i, fmt_order::keep),
    REG_SR(bin, any, bin, OhIw16o32i, fmt_order::keep),

    REG_SR(f32, any, s8, hwio_s8s8, fmt_order::keep),
    REG_SR(f32, any, s8, hwigo_s8s8, fmt_order::keep),
    REG_SR(s8, any, s8, hwio_s8s8, fmt_order::keep),
    REG_SR(s8, any, s8, hwigo_s8s8, fmt_order::keep),

    REG_SR(f32, goihw, s8, gOIhw4o4i_s8s8, fmt_order::keep),
    REG_SR(f32, hwigo, s8, gOIhw4o4i_s8s8, fmt_order::keep),
    REG_SR(s8, goihw, s8, gOIhw4o4i_s8s8, fmt_order::keep),
    REG_SR(s8, hwigo, s8, gOIhw4o4i_s8s8, fmt_order::keep),

    REG_SR(f32, oiw, s8, OIw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(f32, goiw, s8, gOIw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(f32, oihw, s8, OIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(f32, goihw, s8, gOIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(f32, hwio, s8, OIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(f32, hwigo, s8, gOIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, oiw, s8, OIw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, goiw, s8, gOIw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, oihw, s8, OIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, goihw, s8, gOIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, hwio, s8, OIhw4i16o4i_s8s8, fmt_order::keep),
    REG_SR(s8, hwigo, s8, gOIhw4i16o4i_s8s8, fmt_order::keep),

    REG_SR(f32, goihw, s8, gOIhw2i8o4i_s8s8, fmt_order::keep),
    REG_SR(f32, hwigo, s8, gOIhw2i8o4i_s8s8, fmt_order::keep),
    REG_SR(s8, goihw, s8, gOIhw2i8o4i_s8s8, fmt_order::keep),
    REG_SR(s8, hwigo, s8, gOIhw2i8o4i_s8s8, fmt_order::keep),

    REG_SR(f32, goiw, s8, Goiw16g_s8s8, fmt_order::keep),
    REG_SR(f32, goihw, s8, Goihw16g_s8s8, fmt_order::keep),
    REG_SR(f32, hwigo, s8, Goihw16g_s8s8, fmt_order::keep),
    REG_SR(s8, goiw, s8, Goiw16g_s8s8, fmt_order::keep),
    REG_SR(s8, goihw, s8, Goihw16g_s8s8, fmt_order::keep),
    REG_SR(s8, hwigo, s8, Goihw16g_s8s8, fmt_order::keep),

    /* bf16 */
    REG_SR_BIDIR(bf16, any, bf16, nChw16c),

    REG_SR(f32, nchw, bf16, nChw16c, fmt_order::keep),
    REG_SR(bf16, nChw16c, f32, nchw, fmt_order::keep),

    REG_SR(f32, oihw, bf16, OIhw8i16o2i, fmt_order::keep),
    REG_SR(f32, oihw, bf16, IOhw8i16o2i, fmt_order::keep),
    REG_SR(f32, goihw, bf16, gOIhw8i16o2i, fmt_order::keep),
    REG_SR(f32, goihw, bf16, gIOhw8i16o2i, fmt_order::keep),
    REG_SR(f32, oihw, bf16, OIhw8o16i2o, fmt_order::keep),
    REG_SR(f32, goihw, bf16, gOIhw8o16i2o, fmt_order::keep),
    REG_SR(f32, oihw, bf16, IOhw8o16i2o, fmt_order::keep),
    REG_SR(f32, goihw, bf16, gIOhw8o16i2o, fmt_order::keep),
    REG_SR(f32, oihw, bf16, OIhw16i16o, fmt_order::keep),
    REG_SR(f32, goihw, bf16, gOIhw16i16o, fmt_order::keep),

    REG_SR(bf16, OIhw16i16o, f32, oihw, fmt_order::keep),
    REG_SR(bf16, gOIhw16i16o, f32, goihw, fmt_order::keep),

    REG_SR(bf16, any, bf16, any, fmt_order::any, spec::reference),
    REG_SR(bf16, any, f32, any, fmt_order::any, spec::reference),
    REG_SR(f32, any, bf16, any, fmt_order::any, spec::reference),

    /* s16 <-> s16 */
    REG_SR_DIRECT_COPY(s16, s16),

    REG_SR_BIDIR(s16, any, s16, OIhw8i16o2i),
    REG_SR_BIDIR(s16, any, s16, gOIhw8i16o2i),
    REG_SR_BIDIR(s16, OIhw8i16o2i, s16, OIhw8o16i2o),
    REG_SR_BIDIR(s16, gOIhw8i16o2i, s16, gOIhw8o16i2o),

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
