// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>

namespace ov {

/**
 * @brief Set current round direction for scoped block.
 *
 * Round direction can be one of:
 * - FE_DOWNWARD
 * - FE_TONEAREST
 * - FE_TOWARDZERO
 * - FE_UPWARD
 * see std <cfenv> header for details.
 */
class RoundGuard {
public:
    RoundGuard(int mode);
    ~RoundGuard();

private:
    int m_prev_round_mode;
};
}  // namespace ov
