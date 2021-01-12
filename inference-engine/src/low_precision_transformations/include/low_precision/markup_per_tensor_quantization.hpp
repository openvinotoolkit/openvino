// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/lpt_visibility.hpp>

#include "common/operation_per_tensor_quantization_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupPerTensorQuantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::MarkupPerTensorQuantization : public ngraph::pass::FunctionPass {
public:
    class PerTensorQuantization {
    public:
        PerTensorQuantization() = default;
        PerTensorQuantization(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const uint64_t version, const std::vector<size_t>& precisions) {
            precisionsByVersion.emplace(version, precisions);
        }

        bool versionIsRequired;
        std::unordered_map<uint64_t, std::vector<size_t>> precisionsByVersion;
    };

    NGRAPH_RTTI_DECLARATION;
    MarkupPerTensorQuantization(const std::vector<OperationPerTensorQuantizationRestriction>& restrictions = {});
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    std::unordered_map<std::string, PerTensorQuantization> restrictionsByOperation;
};
