// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class InsertSpecificIterations : public Pass {
public:
    OPENVINO_RTTI("InsertSpecificIterations", "Pass")
    bool run(LinearIR& linear_ir) override;

    static LinearIR::container copy_loop(const LinearIR& linear_ir, const size_t loop_id);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
