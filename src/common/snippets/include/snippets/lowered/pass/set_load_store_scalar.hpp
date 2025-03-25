// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SetLoadStoreScalar
 * @brief The pass set scalar count to Load and Store if processing dimension is equal to 1.
 *        Note: works on static shapes
 * @ingroup snippets
 */
class SetLoadStoreScalar : public RangedPass {
public:
    OPENVINO_RTTI("SetLoadStoreScalar", "", RangedPass)
    SetLoadStoreScalar() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
