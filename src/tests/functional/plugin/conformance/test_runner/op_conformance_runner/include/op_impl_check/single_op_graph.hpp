// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace test {
namespace op_conformance {

using OpGenerator = std::map<ov::DiscreteTypeInfo, std::function<std::shared_ptr<ov::Model>()>>;
OpGenerator getOpGeneratorMap();

static const std::vector<std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>> createFunctions() {
    std::vector<std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>> res;
    auto opsets = ov::get_available_opsets();
    auto opGenerator = getOpGeneratorMap();
    std::set<ov::NodeTypeInfo> opsInfo;
    for (const auto& opset_pair : opsets) {
        std::string opset_version = opset_pair.first;
        const ov::OpSet& opset = opset_pair.second();
        const auto &type_info_set = opset.get_type_info_set();
        opsInfo.insert(type_info_set.begin(), type_info_set.end());
    }

    for (const auto& type_info : opsInfo) {
        if (opGenerator.find(type_info) != opGenerator.end())
            res.push_back({type_info, opGenerator.find(type_info)->second()});
    }
    return res;
}

}  // namespace op_conformance
}  // namespace test
}  // namespace ov
