// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @file contains general purpose utils without any MKLDNN or Neural Network inference specific
 */

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline T rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

