// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/common_util.hpp"

#include <algorithm>

std::string ov::util::to_lower(const std::string& s) {
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

std::string ov::util::to_upper(const std::string& s) {
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::toupper);
    return rc;
}

std::vector<std::string> ov::util::split(const std::string& src, char delimiter, bool do_trim) {
    size_t pos;
    std::string token;
    size_t start = 0;
    std::vector<std::string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos) {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim) {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size()) {
        token = src.substr(start);
        if (do_trim) {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

size_t ov::util::hash_combine(const std::vector<size_t>& list) {
    size_t seed = 0;
    for (size_t v : list) {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
