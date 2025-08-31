// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {

void operator>>(const std::stringstream& in, ov::element::Type& type) {
    type = ov::element::Type(ov::util::trim(in.str()));
}

}  // namespace ov
