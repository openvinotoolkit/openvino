// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl.h"

#include <dnnl_debug.h>

#include <cassert>
#include <cpu/platform.hpp>
#include <cstring>

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl::utils {

const char* fmt2str(memory::format_tag fmt) {
    return dnnl_fmt_tag2str(static_cast<dnnl_format_tag_t>(fmt));
}

dnnl::memory::format_tag str2fmt(const char* str) {
#define CASE(_fmt)                                                     \
    do {                                                               \
        if (!strcmp(#_fmt, str) || !strcmp("dnnl_" #_fmt, str))        \
            return static_cast<dnnl::memory::format_tag>(dnnl_##_fmt); \
    } while (0)
    CASE(x);
    CASE(nc);
    CASE(ncw);
    CASE(nwc);
    CASE(nCw4c);
    CASE(nCw8c);
    CASE(nCw16c);
    CASE(nchw);
    CASE(nhwc);
    CASE(chwn);
    CASE(nChw4c);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(oi);
    CASE(io);
    CASE(oiw);
    CASE(wio);
    CASE(OIw16i16o);
    CASE(OIw16o16i);
    CASE(Oiw16o);
    CASE(Owi16o);
    CASE(OIw8i16o2i);
    CASE(OIw4i16o4i);
    CASE(oihw);
    CASE(ihwo);
    CASE(hwio);
    CASE(iohw);
    CASE(dhwio);
    CASE(OIhw8i8o);
    CASE(OIhw16i16o);
    CASE(OIhw8i16o2i);
    CASE(OIdhw8i16o2i);
    CASE(OIhw4i16o4i);
    CASE(OIdhw4i16o4i);
    CASE(OIhw8o16i2o);
    CASE(IOhw8o16i2o);
    CASE(OIhw8o8i);
    CASE(OIhw8o32i);
    CASE(OIhw16o32i);
    CASE(OIhw16o16i);
    CASE(IOhw16o16i);
    CASE(Oihw16o);
    CASE(Ohwi8o);
    CASE(Ohwi16o);
    CASE(goiw);
    CASE(goihw);
    CASE(hwigo);
    CASE(giohw);
    CASE(dhwigo);
    CASE(goiw);
    CASE(gOIw16i16o);
    CASE(gOIw16o16i);
    CASE(gOiw16o);
    CASE(gOwi16o);
    CASE(gOIw8i16o2i);
    CASE(gOIw4i16o4i);
    CASE(Goiw16g);
    CASE(gOIhw8i8o);
    CASE(gOIhw16i16o);
    CASE(gOIhw8i16o2i);
    CASE(gOIdhw8i16o2i);
    CASE(gOIhw2i8o4i);
    CASE(gOIhw4i16o4i);
    CASE(gOIdhw4i16o4i);
    CASE(gOIhw8o16i2o);
    CASE(gIOhw8o16i2o);
    CASE(gOIhw4o4i);
    CASE(gOIhw8o8i);
    CASE(gOIhw16o16i);
    CASE(gIOhw16o16i);
    CASE(gOihw16o);
    CASE(gOhwi8o);
    CASE(gOhwi16o);
    CASE(Goihw8g);
    CASE(Goihw16g);
    CASE(Goidhw4g);
    CASE(Goidhw8g);
    CASE(Goidhw16g);
    CASE(ncdhw);
    CASE(ndhwc);
    CASE(oidhw);
    CASE(goidhw);
    CASE(nCdhw4c);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
    CASE(OIdhw16i16o);
    CASE(gOIdhw16i16o);
    CASE(OIdhw16o16i);
    CASE(gOIdhw16o16i);
    CASE(Oidhw16o);
    CASE(Odhwi16o);
    CASE(gOidhw16o);
    CASE(gOdhwi16o);
    CASE(ntc);
    CASE(tnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldgo);
#undef CASE
    assert(!"unknown memory format");
    return dnnl::memory::format_tag::undef;
}

unsigned get_cache_size(int level, bool per_core) {
    if (per_core) {
        return dnnl::impl::cpu::platform::get_per_core_cache_size(level);
    }
    using namespace dnnl::impl::cpu::x64;
    if (cpu().getDataCacheLevels() == 0) {
        // this function can return stub values in case of unknown CPU type
        return dnnl::impl::cpu::platform::get_per_core_cache_size(level);
    }

    if (level > 0 && static_cast<unsigned>(level) <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l);
    }
    return 0U;
}

}  // namespace dnnl::utils
