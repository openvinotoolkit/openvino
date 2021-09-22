// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
void operator>>(const std::stringstream& in, ngraph::element::Type& type) {
    type = InferenceEngine::details::convertPrecision(ngraph::trim(in.str()));
}

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value) {
    if (!node)
        return false;

    auto attr = node.attribute(name.c_str());
    if (attr.empty())
        return false;
    value = std::string(attr.value());
    return true;
}
}  // namespace ov