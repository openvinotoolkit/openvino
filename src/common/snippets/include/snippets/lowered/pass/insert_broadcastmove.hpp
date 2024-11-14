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
 * @interface InsertMovebroadcast
 * @brief Injects explicit Movebroadcast operations when the most varying dim is broadcasted
 * @ingroup snippets
 */
class InsertBroadcastMove : public RangedPass {
public:
    OPENVINO_RTTI("InsertBroadcastMove", "RangedPass")
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    static bool is_broadcasting_supported(const std::shared_ptr<ov::Node>& n);
private:
    static bool is_broadcasting_needed(const std::shared_ptr<ov::Node>& n);
    static std::vector<size_t> get_last_dims(const ExpressionPtr& expr);
    static size_t get_max_dim(const std::vector<size_t>& last_dims);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
