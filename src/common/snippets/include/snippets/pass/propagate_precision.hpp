// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/pass/pass.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @class PropagatePrecision
 * @ingroup snippets
 * @brief PropagatePrecision transformation propagate precision from parameters to results.
 */
class PropagatePrecision: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::PropagatePrecision");
    PropagatePrecision(const std::shared_ptr<const TargetMachine>& target_machine);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static std::vector<element::Type> get_precisions(const std::vector<element::Type>& input_precisions,
                                                     const std::set<std::vector<element::Type>>& supported_precisions);

    // if can_be_removed returns true then actual convertion (actual_before => actual_after)
    // can be replaced to required (actual_before => required_after)
    static bool can_be_removed(const element::Type& actual_before,
                               const element::Type& actual_after,
                               const element::Type& required_after);

    // if can_be_fused returns true then actual convertion can be replaced to required
    static bool can_be_fused(const element::Type& actual, const element::Type& required);

    static bool validate_and_infer_types_and_restore_outputs(const std::shared_ptr<ov::Node>& op);

private:
    const std::shared_ptr<const TargetMachine> target_machine;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
