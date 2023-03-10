// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/exception.hpp"

namespace ov {

void throw_cancelled(const CheckLocInfo& check_loc_info,
                     const std::string& context_info,
                     const std::string& explanation) {
    throw ov::Cancelled(explanation);
}
void throw_busy(const CheckLocInfo& check_loc_info, const std::string& context_info, const std::string& explanation) {
    throw ov::Busy(explanation);
}

}  // namespace ov
