// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <vector>

#include <ngraph/pass/pass.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/common/operation_precision_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupPrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Transformation is used to add customization options runtime
class ngraph::pass::low_precision::MarkupPrecisions : public ngraph::pass::FunctionPass {
public:
    class Restriction {
    public:
        explicit Restriction(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const uint64_t version, const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisions) {
            precisionsByVersion.emplace(version, precisions);
        }

        bool versionIsRequired;
        std::unordered_map<uint64_t, std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>> precisionsByVersion;
    };

    NGRAPH_RTTI_DECLARATION;
    explicit MarkupPrecisions(const std::vector<OperationPrecisionRestriction>& restrictions = {});
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    static bool isPrecisionPreserved(const std::shared_ptr<Node>& node);
    static bool isSupported(const std::shared_ptr<Node>& node);
    std::unordered_map<std::string, Restriction> restrictionsByOperation;
};
