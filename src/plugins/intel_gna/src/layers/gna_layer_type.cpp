// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_layer_type.hpp"

#include <legacy/graph_tools.hpp>
#include <string>
#include <unordered_set>

#include "gna_layer_info.hpp"

namespace ov {
namespace intel_gna {

LayerType LayerTypeFromStr(const std::string& str) {
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return LayerType::NO_TYPE;
}

}  // namespace intel_gna
}  // namespace ov
