// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/io.hpp>

#include <ostream>

namespace vpu {

void formatPrint(std::ostream& os, const char* str) {
    while (*str) {
        if (*str == '%') {
            if (*(str + 1) == '%') {
                ++str;
            } else {
                std::cerr << "[VPU] Invalid format string : missing arguments" << std::endl;
                return;
            }
        } else if (*str == '{') {
            if (*(str + 1) == '}') {
                std::cerr << "[VPU] Invalid format string : missing arguments" << std::endl;
                return;
            }
        }

        os << *str++;
    }
}

}  // namespace vpu
