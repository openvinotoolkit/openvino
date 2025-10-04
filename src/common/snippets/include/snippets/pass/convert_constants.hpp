// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace only constants which are should be represented as scalars during code generation.
 *        Only single-value (0D) constants are currently supported.
 * @ingroup snippets
 */
class ConvertConstantsToScalars : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::ConvertConstantsToScalars");
    ConvertConstantsToScalars() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::snippets::pass
