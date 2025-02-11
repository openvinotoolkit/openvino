// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "snippets/generator.hpp"

namespace ov::intel_cpu::pass {

class EnforcePrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("EnforcePrecision");

    EnforcePrecision(const element::Type source,
                     const element::Type target,
                     const std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)>&
                         get_supported_precisions = nullptr);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    static std::set<std::vector<element::Type>> get_supported_precisions_default(
        const std::shared_ptr<ov::Node>& op) noexcept;

    const element::Type source;
    const element::Type target;
    const std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)>
        get_supported_precisions;
};

}  // namespace ov::intel_cpu::pass
