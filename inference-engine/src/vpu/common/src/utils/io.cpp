// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/io.hpp>

#include <iostream>

#include <vpu/utils/any.hpp>
#include <vpu/utils/attributes_map.hpp>

namespace vpu {

void printTo(std::ostream& os, const Any& any) noexcept {
    any.printImpl(os);
}

void printTo(std::ostream& os, const AttributesMap& attrs) noexcept {
    attrs.printImpl(os);
}

void formatPrint(std::ostream& os, const char* str) noexcept {
    try {
        while (*str) {
            if (*str == '%') {
                if (*(str + 1) == '%') {
                    ++str;
                } else {
                    throw std::invalid_argument("[VPU] Invalid format string : missing arguments");
                }
            }

            os << *str++;
        }
    } catch (std::invalid_argument e) {
        std::cerr << e.what() << '\n';
        std::abort();
    } catch (...) {
        std::cerr << "[VPU] Unknown error in formatPrint\n";
        std::abort();
    }
}

}  // namespace vpu
