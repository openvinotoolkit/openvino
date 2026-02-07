// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/weights_path.hpp"

bool ov::util::validate_weights_path(const std::string& weights_path) {
    if (weights_path.empty() || !ov::util::ends_with(weights_path, ".bin")) {
        return false;
    }

    return true;
}
