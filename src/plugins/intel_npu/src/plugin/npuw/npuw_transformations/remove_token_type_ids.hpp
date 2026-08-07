// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov::npuw {

constexpr const char* token_type_ids_name = "token_type_ids";

class RemoveTokenTypeIds : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RemoveTokenTypeIds");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace ov::npuw
