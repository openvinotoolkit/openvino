// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {

void operator>>(const std::stringstream& in, ov::element::Type& type) {
    const std::string type_name{util::trim(in.str())};
    type = element::Type(type_name);
}

}  // namespace ov
