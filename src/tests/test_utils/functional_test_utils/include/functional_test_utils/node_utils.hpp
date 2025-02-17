// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "openvino/opsets/opset.hpp"

namespace ov {
namespace test {
namespace utils {

inline std::string get_op_version(std::string version_full_name) {
    std::string op_version(version_full_name);
    std::string opset_name = "opset";

    auto pos = op_version.find(opset_name);
    if (pos != std::string::npos) {
        op_version = op_version.substr(pos + opset_name.size());
    }

    return op_version;
}

static std::map<std::string, std::set<std::string>> get_unique_ops() {
    // { op_name, { opsets }}
    std::map<std::string, std::set<std::string>> res;
    for (const auto& opset_pair : ov::get_available_opsets()) {
        std::string opset_name = opset_pair.first;
        const ov::OpSet& opset = opset_pair.second();
        for (const auto& op : opset.get_type_info_set()) {
            std::string op_version = get_op_version(op.get_version());
            if (res.find(op.name) == res.end()) {
                res.insert({op.name, {}});
            }
            res[op.name].insert(op_version);
        }
    }
    return res;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
