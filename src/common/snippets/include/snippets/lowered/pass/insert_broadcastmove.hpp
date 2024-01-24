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
class InsertBroadcastMove : public Pass {
public:
    OPENVINO_RTTI("InsertBroadcastMove", "Pass")
    bool run(LinearIR& linear_ir) override;

private:
    static bool is_broadcasting_supported(const std::shared_ptr<ov::Node>& n);
    static bool is_broadcasting_needed(const std::shared_ptr<ov::Node>& n);
    static std::vector<size_t> get_last_dims(const ExpressionPtr& expr);
    static size_t get_broadcast_dim(const std::vector<size_t>& last_dims);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
