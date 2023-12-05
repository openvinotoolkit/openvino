// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/generator.hpp"
#include "openvino/pass/graph_rewrite.hpp"

#include <memory>

namespace ov {
namespace intel_cpu {
namespace pass {

class EnforcePrecision: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("EnforcePrecision", "0");

    EnforcePrecision(
        const element::Type source,
        const element::Type target,
        std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)> get_supported_precisions = nullptr);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    static std::set<std::vector<element::Type>> get_supported_precisions_default(const std::shared_ptr<ov::Node>& op) noexcept;

    const element::Type source;
    const element::Type target;
    const std::function<std::set<std::vector<element::Type>>(const std::shared_ptr<ov::Node>& op)> get_supported_precisions;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
