// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(HAVE_SSE) || defined(HAVE_AVX2)
#    if defined(_WIN32)
#        include <emmintrin.h>
#    else
#        include <x86intrin.h>
#    endif
#endif
