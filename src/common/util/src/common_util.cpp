// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/common_util.hpp"

#include <algorithm>
#include <iterator>

std::string ov::util::to_lower(std::string_view s) {
    std::string rc{s};
    std::transform(rc.begin(), rc.end(), rc.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return rc;
}

std::string ov::util::to_upper(std::string_view s) {
    std::string rc{s};
    std::transform(rc.begin(), rc.end(), rc.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    return rc;
}

std::string ov::util::filter_lines_by_prefix(std::string_view sv, std::string_view prefix) {
    std::ostringstream res;
    view_transform_if(sv, std::ostream_iterator<std::string_view>(res, "\n"), "\n", [&prefix](auto&& field) {
        return field.find(prefix) == 0;
    });
    return res.str();
}
