// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <sstream>
#include <utility>

#include <details/caseless.hpp>

#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

namespace details {

inline void insertToContainer(std::vector<std::string>& cont, std::string&& val) {
    cont.emplace_back(std::move(val));
}

template <int Capacity>
void insertToContainer(SmallVector<std::string, Capacity>& cont, std::string&& val) {
    cont.emplace_back(std::move(val));
}

inline void insertToContainer(std::set<std::string>& cont, std::string&& val) {
    cont.emplace(std::move(val));
}

inline void insertToContainer(std::unordered_set<std::string>& cont, std::string&& val) {
    cont.emplace(std::move(val));
}

inline void insertToContainer(ie::details::caseless_set<std::string>& cont, std::string&& val) {
    cont.emplace(std::move(val));
}

}  // namespace details

template <class Cont>
void splitStringList(const std::string& str, Cont& out, char delim) {
    out.clear();

    if (str.empty())
        return;

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }

        details::insertToContainer(out, std::move(elem));
    }
}

template <class Cont>
Cont splitStringList(const std::string& str, char delim) {
    Cont out;
    splitStringList(str, out, delim);
    return out;
}

}  // namespace vpu
