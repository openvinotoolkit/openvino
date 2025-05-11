// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
// Todo: Replace remarks with DEBUG_CAPS
class logstreambuf: public std::streambuf {
public:
    static const int threshold {50};
};

template <typename T>
static inline auto remark(T x) -> std::ostream& {
    static logstreambuf nostreambuf;
    static std::ostream nocout(&nostreambuf);

    return ((x >= logstreambuf::threshold)? std::cout << "Remark: " : nocout);
}
