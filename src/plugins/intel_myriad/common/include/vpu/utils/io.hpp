// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/small_vector.hpp>
#include <vpu/utils/intrusive_handle_list.hpp>

#include <utility>
#include <string>
#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>

namespace vpu {

//
// printTo
//

//
// Controls the format printing for actual type
//

template <typename T>
void printTo(std::ostream& os, const T& val);

template <typename T1, typename T2>
void printTo(std::ostream& os, const std::pair<T1, T2>& p);

template <typename T, size_t Count>
void printTo(std::ostream& os, const std::array<T, Count>& cont);

template <typename T, class A>
void printTo(std::ostream& os, const std::vector<T, A>& cont);

template <typename T, int Capacity, class A>
void printTo(std::ostream& os, const SmallVector<T, Capacity, A>& cont);

template <typename T, class A>
void printTo(std::ostream& os, const std::list<T, A>& cont);

template <class Obj>
void printTo(std::ostream& os, const IntrusiveHandleList<Obj>& cont);

template <typename T, class C, class A>
void printTo(std::ostream& os, const std::set<T, C, A>& cont);

template <typename T, class H, class P, class A>
void printTo(std::ostream& os, const std::unordered_set<T, H, P, A>& cont);

template <typename K, typename V, class C, class A>
void printTo(std::ostream& os, const std::map<K, V, C, A>& map);

template <typename K, typename V, class H, class P, class A>
void printTo(std::ostream& os, const std::unordered_map<K, V, H, P, A>& map);

//
// formatPrint
//

//
// The format printing supports with following placeholders:
//
//   * C like : `%?`, where `?` is any character except `%`, `%%` will be converted to single `%` symbol.
//   * Python like : `{}`.
//
// The formating manipulators are not supported in the formating string placeholders.
// Instead, the manipulators can be added as plain arguments with own placeholder, for example:
//
//     formatPrint(os, "The is the number with formatting: {}{}", std::setw(5), 100);
//

void formatPrint(std::ostream& os, const char* str);

template <typename T, typename... Args>
void formatPrint(std::ostream& os, const char* str, const T& val, const Args&... args);

//
// formatString
//

template <typename... Args>
std::string formatString(const char* str, const Args&... args) {
    std::ostringstream os;
    formatPrint(os, str, args...);
    return os.str();
}

//
// toString
//

template <typename T>
std::string toString(const T& val) {
    std::ostringstream os;
    printTo(os, val);
    return os.str();
}

//
// Implementation
//

namespace details {

template <typename T>
auto printToDefault(std::ostream& os, const T& val, int) -> decltype(os << val) {
    return os << val;
}
template <typename T>
void printToDefault(std::ostream&, const T&, ...) {
    // Nothing
}

}  // namespace details

template <typename T>
void printTo(std::ostream& os, const T& val) {
    details::printToDefault(os, val, 0);
}

template <typename T1, typename T2>
void printTo(std::ostream& os, const std::pair<T1, T2>& p) {
    os << '(';
    printTo(os, p.first);
    os << ", ";
    printTo(os, p.second);
    os << ')';
}

namespace details {

template <class Cont>
void printContainer(std::ostream& os, const Cont& cont) {
    using IndexType = decltype(cont.size());
    static constexpr IndexType MAX_PRINT_SIZE = 8;

    os << '[';

    IndexType ind = 0;
    for (const auto& val : cont) {
        printTo(os, val);

        if (ind + 1 < cont.size()) {
            os << ", ";
        }

        if (ind > MAX_PRINT_SIZE) {
            os << "...";
            break;
        }

        ++ind;
    }

    os << ']';
}

}  // namespace details

template <typename T, size_t Count>
void printTo(std::ostream& os, const std::array<T, Count>& cont) {
    details::printContainer(os, cont);
}

template <typename T, class A>
void printTo(std::ostream& os, const std::vector<T, A>& cont) {
    details::printContainer(os, cont);
}

template <typename T, int Capacity, class A>
void printTo(std::ostream& os, const SmallVector<T, Capacity, A>& cont) {
    details::printContainer(os, cont);
}

template <typename T, class A>
void printTo(std::ostream& os, const std::list<T, A>& cont) {
    details::printContainer(os, cont);
}

template <class Obj>
void printTo(std::ostream& os, const IntrusiveHandleList<Obj>& cont) {
    details::printContainer(os, cont);
}

template <typename T, class C, class A>
void printTo(std::ostream& os, const std::set<T, C, A>& cont) {
    details::printContainer(os, cont);
}

template <typename T, class H, class P, class A>
void printTo(std::ostream& os, const std::unordered_set<T, H, P, A>& cont) {
    details::printContainer(os, cont);
}

namespace details {

template <class Map>
void printMap(std::ostream& os, const Map& map) {
    static constexpr size_t MAX_PRINT_SIZE = 8;

    os << '[';

    size_t ind = 0;
    for (const auto& p : map) {
        printTo(os, p.first);
        os << ':';
        printTo(os, p.second);

        if (ind + 1 < map.size()) {
            os << ", ";
        }

        if (ind > MAX_PRINT_SIZE) {
            os << "...";
            break;
        }

        ++ind;
    }

    os << ']';
}

}  // namespace details

template <typename K, typename V, class C, class A>
void printTo(std::ostream& os, const std::map<K, V, C, A>& map) {
    details::printMap(os, map);
}

template <typename K, typename V, class H, class P, class A>
void printTo(std::ostream& os, const std::unordered_map<K, V, H, P, A>& map) {
    details::printMap(os, map);
}

template <typename T, typename... Args>
void formatPrint(std::ostream& os, const char* str, const T& val, const Args&... args) {
    while (*str) {
        if (*str == '%') {
            if (*(str + 1) == '%') {
                ++str;
            } else {
                printTo(os, val);
                formatPrint(os, str + 2, args...);
                return;
            }
        } else if (*str == '{') {
            if (*(str + 1) == '}') {
                printTo(os, val);
                formatPrint(os, str + 2, args...);
                return;
            }
        }

        os << *str++;
    }

    std::cerr << "[VPU] Extra arguments provided to formatPrint\n";
}

}  // namespace vpu
