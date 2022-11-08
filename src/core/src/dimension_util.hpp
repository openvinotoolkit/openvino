// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/util/common_util.hpp>
#include <string>
#include <sstream>

namespace ov {
namespace util {

namespace {
bool check_all_digits(const std::string &value) {
    auto val = ov::util::trim(value);
    for (const auto &c: val) {
        if (!std::isdigit(c) || c == '-')
            return false;
    }
    return true;
}
}

}
}