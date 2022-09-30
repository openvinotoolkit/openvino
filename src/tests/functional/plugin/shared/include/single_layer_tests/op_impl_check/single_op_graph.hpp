// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional_test_utils/summary/op_summary.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

namespace ov {
namespace test {
namespace subgraph {

using OpGenerator = std::map<ov::DiscreteTypeInfo, std::function<std::shared_ptr<ov::Model>()>>;
OpGenerator getOpGeneratorMap();

static const std::vector<std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>> createFunctions() {
    std::vector<std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>>> res;
    auto opsets = ov::test::utils::OpSummary::getInstance().getOpSets();
    auto opGenerator = getOpGeneratorMap();
    std::set<ngraph::NodeTypeInfo> opsInfo;
    for (const auto& opset : opsets) {
        const auto &type_info_set = opset.get_type_info_set();
        opsInfo.insert(type_info_set.begin(), type_info_set.end());
    }

    for (const auto& type_info : opsInfo) {
        if (opGenerator.find(type_info) != opGenerator.end())
            res.push_back({type_info, opGenerator.find(type_info)->second()});
    }
    return res;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
