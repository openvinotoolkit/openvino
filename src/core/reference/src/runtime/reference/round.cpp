// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/round.hpp"

namespace ov {
namespace reference {

void set_round_mode(const op::v5::Round::RoundMode& mode) {
    if (mode == op::v5::Round::RoundMode::HALF_TO_EVEN) {
        std::fesetround(FE_TONEAREST);
    }
}
}  // namespace reference
}  // namespace ov
