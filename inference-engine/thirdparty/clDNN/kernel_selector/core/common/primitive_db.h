/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
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
