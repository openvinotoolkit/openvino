// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <string>
#include <array>

#include <ie_data.h>
#include <ie_blob.h>
#include <ie_layers.h>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/containers.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// printTo
//

template <typename T>
void printTo(std::ostream& os, const T& val) noexcept;

template <typename T1, typename T2>
void printTo(std::ostream& os, const std::pair<T1, T2>& p) noexcept;

template <typename T>
void printTo(std::ostream& os, const std::vector<T>& cont) noexcept;

template <typename T, size_t COUNT>
void printTo(std::ostream& os, const std::array<T, COUNT>& cont) noexcept;

template <typename T>
void printTo(std::ostream& os, const std::set<T>& cont) noexcept;

template <typename T, class H>
void printTo(std::ostream& os, const std::unordered_set<T, H>& cont) noexcept;

template <typename K, typename V>
void printTo(std::ostream& os, const std::map<K, V>& map) noexcept;

template <typename K, typename V, class H>
void printTo(std::ostream& os, const std::unordered_map<K, V, H>& map) noexcept;

template <typename T, int Capacity>
void printTo(std::ostream& os, const SmallVector<T, Capacity>& cont) noexcept;

class Any;
void printTo(std::ostream& os, const Any& any) noexcept;

class AttributesMap;
void printTo(std::ostream& os, const AttributesMap& attrs) noexcept;

//
// formatPrint
//

void formatPrint(std::ostream& os, const char* str) noexcept;

template <typename T, typename... Args>
void formatPrint(std::ostream& os, const char* str, const T& value, const Args&... args) noexcept;

//
// formatString
//

template <typename... Args>
std::string formatString(const char* str, const Args&... args) noexcept;

//
// toString
//

template <typename T>
std::string toString(const T& val) noexcept;

//
// Implementation
//

template <typename T>
void printTo(std::ostream& os, const T& val) noexcept {
    try {
        os << val;
    } catch (...) {
        std::cerr << "[VPU] Unknown error while printing\n";
        std::abort();
    }
}

template <typename T1, typename T2>
void printTo(std::ostream& os, const std::pair<T1, T2>& p) noexcept {
    try {
        os << "[" << std::endl;

        os << "first=";
        printTo(os, p.first);
        os << std::endl;

        os << "second=";
        printTo(os, p.second);
        os << std::endl;

        os << "]";
    } catch (...) {
        std::cerr << "[VPU] Unknown error while printing\n";
        std::abort();
    }
}

template <class Cont>
void printContainer(std::ostream& os, const Cont& cont) noexcept {
    try {
        os << "[";

        size_t ind = 0;
        for (const auto& val : cont) {
            printTo(os, val);
            if (ind + 1 < cont.size()) {
                os << ", ";
            }
            if (ind > 8) {
                os << "...";
                break;
            }
            ++ind;
        }

        os << "]";
    } catch (...) {
        std::cerr << "[VPU] Unknown error while printing\n";
        std::abort();
    }
}

template <typename T>
void printTo(std::ostream& os, const std::vector<T>& cont) noexcept {
    printContainer(os, cont);
}

template <typename T, size_t COUNT>
void printTo(std::ostream& os, const std::array<T, COUNT>& cont) noexcept {
    printContainer(os, cont);
}

template <typename T>
void printTo(std::ostream& os, const std::set<T>& cont) noexcept {
    printContainer(os, cont);
}

template <typename T, class H>
void printTo(std::ostream& os, const std::unordered_set<T, H>& cont) noexcept {
    printContainer(os, cont);
}

template <class Map>
void printMap(std::ostream& os, const Map& map) noexcept {
    try {
        os << "[" << std::endl;

        size_t ind = 0;
        for (const auto& p : map) {
            printTo(os, p.first);
            os << "=";
            printTo(os, p.second);
            os << std::endl;
            if (ind > 16) {
                os << "...";
                break;
            }
            ++ind;
        }

        os << "]";
    } catch (...) {
        std::cerr << "[VPU] Unknown error while printing\n";
        std::abort();
    }
}

template <typename K, typename V>
void printTo(std::ostream& os, const std::map<K, V>& map) noexcept {
    printMap(os, map);
}

template <typename K, typename V, class H>
void printTo(std::ostream& os, const std::unordered_map<K, V, H>& map) noexcept {
    printMap(os, map);
}

template <typename T, int Capacity>
void printTo(std::ostream& os, const SmallVector<T, Capacity>& cont) noexcept {
    printContainer(os, cont);
}

template <typename T, typename... Args>
void formatPrint(std::ostream& os, const char* str, const T& value, const Args&... args) noexcept {
    try {
        while (*str) {
            if (*str == '%') {
                if (*(str + 1) == '%') {
                    ++str;
                } else {
                    printTo(os, value);
                    formatPrint(os, str + 2, args...);
                    return;
                }
            }

            os << *str++;
        }
    } catch (...) {
        std::cerr << "[VPU] Unknown error while printing\n";
        std::abort();
    }

    std::cerr << "[VPU] Extra arguments provided to formatPrint\n";
    std::abort();
}

template <typename T>
std::string toString(const T& val) noexcept {
    std::ostringstream os;
    printTo(os, val);
    return os.str();
}

template <typename... Args>
std::string formatString(const char* str, const Args&... args) noexcept {
    std::ostringstream os;
    formatPrint(os, str, args...);
    return os.str();
}

}  // namespace vpu
