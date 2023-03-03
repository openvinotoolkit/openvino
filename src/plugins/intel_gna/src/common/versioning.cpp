// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "versioning.hpp"

#include <sstream>
#include <string>

#include "openvino/core/version.hpp"

namespace ov {
namespace intel_gna {
namespace common {

std::string get_openvino_version_string() {
    std::stringstream s;
    s << ov::get_openvino_version();
    return s.str();
}
} // namespace common
} // namespace intel_gna
} // namespace ov
