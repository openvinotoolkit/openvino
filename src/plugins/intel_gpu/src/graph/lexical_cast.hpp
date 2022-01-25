// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <json_object.h>

namespace cldnn {
inline std::string lexical_cast(const json_base& j, int offset = 1) {
    std::stringstream os;
    j.dump(os, offset);
    return os.str();
}

}  // namespace cldnn


