// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline int FUNC(get_nearest_val)(float num, bool is_downsample)
{
#if defined(NEAREST_ROUND_PREFER_FLOOR)
    return (num == (int)num + 0.5f) ? (int)floor(num) : (int)round(num);
#elif defined(NEAREST_ROUND_PREFER_CEIL)
    return (int)round(num);
#elif defined(NEAREST_FLOOR)
    return (int)floor(num);
#elif defined(NEAREST_CEIL)
    return (int)ceil(num);
#elif defined(NEAREST_SIMPLE)
    return is_downsample ? (int)ceil(num) : (int)num;
#else
#error [clDNN resample]: nearest mode - not supported
#endif
}
