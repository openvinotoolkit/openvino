// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace snippets {

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
class FunctionHelper {
public:
    static std::vector<std::shared_ptr<Node>> makePrerequisitesOriginal();

    static std::shared_ptr<Node> applyPrerequisites(
        const std::shared_ptr<Node>& parent,
        const std::vector<std::shared_ptr<Node>>& prerequisites);

    // index: -1 - latest `Subgraph` operation
    static std::shared_ptr<Node> getSubgraph(const std::shared_ptr<Model>& f, const int index = -1);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
