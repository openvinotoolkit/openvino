// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <cctype>
#include <string>

/// \brief Class providing interface to retrieve a list of primitive implementations per primitive id
///
namespace kernel_selector {
namespace gpu {
namespace cache {

using code = std::string;
using primitive_id = std::string;

struct primitive_db {
    primitive_db();

    std::vector<code> get(const primitive_id& id) const;

private:
    struct case_insensitive_compare {
        bool operator()(const primitive_id& lhs, const primitive_id& rhs) const {
            return std::lexicographical_compare(lhs.begin(),
                                                lhs.end(),
                                                rhs.begin(),
                                                rhs.end(),
                                                [](const char& a, const char& b) { return tolower(a) < tolower(b); });
        }
    };
    std::multimap<primitive_id, code, case_insensitive_compare> primitives;
};

}  // namespace cache
}  // namespace gpu
}  // namespace kernel_selector
