// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface TransformConvertToConvertTruncation
 * @brief Transform Convert to ConvertTruncation with specification conversion rules
 *        Note: ConvertTruncation op is covered by specification of "Convert" op
 *              This op is used for real Convert ops inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class TransformConvertToConvertTruncation : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::TransformConvertToConvertTruncation");
    TransformConvertToConvertTruncation() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::snippets::pass
