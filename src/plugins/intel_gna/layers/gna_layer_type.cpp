// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <unordered_set>
#include <legacy/graph_tools.hpp>
#include "gna_layer_type.hpp"
#include "gna_layer_info.hpp"

GNAPluginNS::LayerType GNAPluginNS::LayerTypeFromStr(const std::string &str) {
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return LayerType::NO_TYPE;
}
