// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "iml_type_mapper.h"

using namespace MKLDNNPlugin;

impl_desc_type MKLDNNPlugin::parse_impl_name(std::string impl_desc_name) {
    impl_desc_type res = impl_desc_type::unknown;

#define REPLACE_WORD(_wrd, _sub) auto pos = impl_desc_name.find(#_wrd); \
    if (pos != std::string::npos) impl_desc_name.replace(pos, std::string(#_wrd).length(), #_sub);

    REPLACE_WORD(simple, ref);
#undef REPLACE_WORD

#define SEARCH_WORD(_wrd) if (impl_desc_name.find(#_wrd) != std::string::npos) \
    res = static_cast<impl_desc_type>(res | impl_desc_type::_wrd);
#define SEARCH_WORD_2(_wrd, _key) if (impl_desc_name.find(#_wrd) != std::string::npos) \
    res = static_cast<impl_desc_type>(res | impl_desc_type::_key);

    SEARCH_WORD(ref);
    SEARCH_WORD(jit);
    SEARCH_WORD(gemm);
    SEARCH_WORD(blas);
    SEARCH_WORD(sse42);
    SEARCH_WORD_2(sse41, sse42);
    SEARCH_WORD(avx2);
    SEARCH_WORD(avx512);
    SEARCH_WORD(any);
    SEARCH_WORD(_1x1);
    SEARCH_WORD(_dw);
    SEARCH_WORD(reorder);
    if ((res & impl_desc_type::avx2) != impl_desc_type::avx2 &&
        (res & impl_desc_type::avx512) != impl_desc_type::avx512)
        SEARCH_WORD(avx);
    if ((res & impl_desc_type::sse42) != impl_desc_type::sse42 &&
        (res & impl_desc_type::avx) != impl_desc_type::avx &&
        (res & impl_desc_type::avx2) != impl_desc_type::avx2 &&
        (res & impl_desc_type::avx512) != impl_desc_type::avx512)
        SEARCH_WORD(uni);

    SEARCH_WORD_2(nchw, ref);
    SEARCH_WORD_2(ncdhw, ref);
    SEARCH_WORD_2(wino, winograd);

#undef SEARCH_WORD_2
#undef SEARCH_WORD

    return res;
}
