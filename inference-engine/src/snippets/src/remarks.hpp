// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

class logstreambuf: public std::streambuf {
public:
    static const int threshold {5};
};

template <typename T>
static inline auto remark(T x) -> std::ostream& {
    static logstreambuf nostreambuf;
    static std::ostream nocout(&nostreambuf);

    return ((x >= logstreambuf::threshold)? std::cout << "Remark: " : nocout);
}
