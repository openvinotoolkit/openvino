//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scenario/inference.hpp"

#include <algorithm>
#include <iterator>

std::vector<std::string> extractLayerNames(const std::vector<LayerInfo>& layers) {
    std::vector<std::string> names;
    std::transform(layers.begin(), layers.end(), std::back_inserter(names), [](const auto& layer) {
        return layer.name;
    });
    return names;
}
