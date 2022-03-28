// Copyright (C) 2018-2022 Intel Corporation
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

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupPerTensorQuantization transformation marks operations as required per-tensor quantization according to the
 * provided restrictions.
 *
 * For more details about the transformation, refer to
 * [MarkupPerTensorQuantization](@ref openvino_docs_OV_UG_lpt_MarkupPerTensorQuantization) page
 * in the Inference Engine Developer Guide.
 */
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
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    std::unordered_map<std::string, PerTensorQuantization> restrictionsByOperation;
};
