// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "transformations/low_precision/quantization_details.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {


class TransformationContext {
public:
    explicit TransformationContext(std::shared_ptr<Function> network);

    // TODO: not needed?
    void removeLayer(std::shared_ptr<Node> layer);

    std::shared_ptr<Function> network;
    std::unordered_set<std::string> quantizedFakeQuantizeNames;
    std::unordered_set<std::string> dequantizationLayersNames;

    // TODO: not needed?
//    const std::vector<std::shared_ptr<Node>>& getLayers() {
//        // WARNING: big vector copying
//        return layers;
//    }

    inline ngraph::element::Type getOriginalLayerPrecision(const std::string& layer_name, const size_t output_index = 0) {
        const auto& data_map = _original_precisions_map.find(layer_name);
        if (data_map == _original_precisions_map.end())
            return element::undefined;
        if (data_map->second.find(output_index) == data_map->second.end())
            return element::undefined;
        return data_map->second[output_index];
    }

private:
    //std::vector<std::shared_ptr<Node>> layers;
    // TODO LPT-TO-NGRAPH: change it (replace unordered_map by a vector if no gaps)?
    std::unordered_map<std::string, std::unordered_map<size_t, ngraph::element::Type>> _original_precisions_map;
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph