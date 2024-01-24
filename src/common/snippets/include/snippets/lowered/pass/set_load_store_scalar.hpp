// Copyright (C) 2023 Intel Corporation
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
 * @ingroup snippets
 */
class SetLoadStoreScalar : public Pass {
public:
    OPENVINO_RTTI("SetLoadStoreScalar", "Pass")
    SetLoadStoreScalar() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
