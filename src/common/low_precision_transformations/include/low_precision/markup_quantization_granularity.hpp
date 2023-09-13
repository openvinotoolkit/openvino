// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/pass/pass.hpp"
#include "low_precision/common/port_quantization_granularity_restriction.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupQuantizationGranularity;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupPerTensorQuantization transformation marks operations as required per-tensor quantization according to the
 * provided restrictions.
 *
 * For more details about the transformation, refer to
 * [MarkupPerTensorQuantization](@ref openvino_docs_OV_UG_lpt_MarkupPerTensorQuantization) page
 * in the Inference Engine Developer Guide.
 */
class ov::pass::low_precision::MarkupQuantizationGranularity : public ov::pass::ModelPass {
public:
    class PerTensorQuantization {
    public:
        explicit PerTensorQuantization(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const std::string version_id, const std::vector<PortQuantizationGranularityRestriction>& restrictions) {
            portsByVersion.emplace(version_id, restrictions);
        }

        bool versionIsRequired;
        std::unordered_map<std::string, std::vector<PortQuantizationGranularityRestriction>> portsByVersion;
    };

    OPENVINO_RTTI("MarkupPerTensorQuantization", "0");
    explicit MarkupQuantizationGranularity(const std::vector<QuantizationGranularityRestriction>& restrictions = {});
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::unordered_map<std::string, PerTensorQuantization> restrictionsByOperation;
};
