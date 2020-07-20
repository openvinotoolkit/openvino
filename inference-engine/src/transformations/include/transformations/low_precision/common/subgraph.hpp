// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/check.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class Subgraph {
public:
    bool fillSubgraphForConcat(ngraph::opset1::Concat& concat, std::unordered_set<std::string>& handledLayers);
    bool empty() const;

    std::vector<ngraph::Node*> quantizationLayers;
    std::vector<ngraph::opset1::Concat*> concatLayers;
    std::unordered_map<std::string, ngraph::Node*> layers;

private:
    bool fillSubgraphForQuantization(ngraph::opset1::FakeQuantize& fakeQuantize, std::unordered_set<std::string>& handledLayers);
    bool fillSubgraphForIntermediate(ngraph::Node& intermediate, std::unordered_set<std::string>& handledLayers);
    bool fill(ngraph::Node& concat, std::unordered_set<std::string>& handledLayers);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
