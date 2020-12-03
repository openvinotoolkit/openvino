// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/check.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "../ilayer_transformations_manager.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class Subgraph {
public:
    Subgraph(ngraph::pass::ILayerTransformationsManager* layerTransformationsManager);

    bool fillSubgraphForConcat(const std::shared_ptr<ngraph::opset1::Concat>& concat, std::unordered_set<std::string>& handledLayers);
    bool empty() const;

    std::vector<std::shared_ptr<ngraph::Node>> quantizationLayers;
    std::vector<std::shared_ptr<ngraph::opset1::Concat>> concatLayers;
    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>> layers;

private:
    bool atLeastOneIsIntermediate(const std::shared_ptr<ngraph::Node>& node) const;
    bool fillSubgraphForQuantization(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fakeQuantize, std::unordered_set<std::string>& handledLayers);
    bool fillSubgraphForIntermediate(const std::shared_ptr<ngraph::Node>& intermediate, std::unordered_set<std::string>& handledLayers);
    bool fill(const std::shared_ptr<ngraph::Node>& concat, std::unordered_set<std::string>& handledLayers);
    const ngraph::pass::ILayerTransformationsManager* layerTransformationsManager;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
