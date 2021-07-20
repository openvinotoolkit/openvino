// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <ngraph/pass/pass.hpp>
#include "common/operation_per_tensor_quantization_restriction.hpp"
#include "low_precision/lpt_visibility.hpp"

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
        explicit PerTensorQuantization(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const uint64_t version, const std::vector<size_t>& ports) {
            portsByVersion.emplace(version, ports);
        }

        bool versionIsRequired;
        std::unordered_map<uint64_t, std::vector<size_t>> portsByVersion;
    };

    NGRAPH_RTTI_DECLARATION;
    explicit MarkupPerTensorQuantization(const std::vector<OperationPerTensorQuantizationRestriction>& restrictions = {});
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    std::unordered_map<std::string, PerTensorQuantization> restrictionsByOperation;
};
