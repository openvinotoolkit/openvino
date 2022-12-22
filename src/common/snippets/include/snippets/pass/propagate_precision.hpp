// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @class PropagatePrecision
 * @ingroup snippets
 * @brief PropagatePrecision transformation propagate precision from parameters to results.
 *
 * PropagatePrecision transformation is one traversal transformation.
 */
class PropagatePrecision: ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("PropagatePrecision", "0");
    PropagatePrecision(
        const ov::element::Type supported_precision,
        const std::shared_ptr<const TargetMachine>& target_machine);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static std::vector<InferenceEngine::Precision> get_precisions(
        const std::vector<InferenceEngine::Precision>& input_precisions,
        const std::set<std::vector<InferenceEngine::Precision>>& supported_precisions,
        const InferenceEngine::Precision& base_precision) noexcept;

private:
    const ov::element::Type supported_precision;
    const std::shared_ptr<const TargetMachine> target_machine;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
