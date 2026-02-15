// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu::pass {

class EnforcePrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("EnforcePrecision");

    EnforcePrecision(element::Type source,
                     element::Type target,
                     const std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)>&
                         get_supported_precisions);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const element::Type source;
    const element::Type target;
    const std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)>
        get_supported_precisions;
};

}  // namespace ov::intel_cpu::pass
