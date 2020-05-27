// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace GNAPluginNS {
/**
 * @brief vector hash not a part of c++11, alternative to use boost hash_combine
 */
class hash_combine_t {
 public:
    size_t operator() (const std::vector<size_t> &input) const {
        // boost hash_combine formula
        size_t seed = 0;
        for (const auto & v : input) {
            seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

}  // namespace GNAPluginNS