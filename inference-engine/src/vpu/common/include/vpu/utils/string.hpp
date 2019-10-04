// Copyright (C) 2018-2019 Intel Corporation
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

#include <vpu/utils/containers.hpp>

namespace vpu {

namespace ie = InferenceEngine;

namespace details {

inline void insertToContainer(std::vector<std::string>& cont, std::string&& val) {
    cont.emplace_back(val);
}

template <int Capacity>
void insertToContainer(SmallVector<std::string, Capacity>& cont, std::string&& val) {
    cont.emplace_back(val);
}

inline void insertToContainer(std::set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
}

inline void insertToContainer(std::unordered_set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
}

inline void insertToContainer(ie::details::caseless_set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
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

}  // namespace vpu
