// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model);

class ConvertKVCacheToPrecision : public ov::pass::ModelPass {
    ov::element::Type m_lp_type;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ConvertKVCacheToPrecision");
    explicit ConvertKVCacheToPrecision(const ov::element::Type lptype);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
