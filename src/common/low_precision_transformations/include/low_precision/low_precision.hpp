// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

// one place to include all Low Precision Transformations from ov::pass::low_precision
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/align_quantization_intervals.hpp"


#include "low_precision/lpt_visibility.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"
#include "low_precision/common/precisions_restriction.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/markup_precisions.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API TypeRelaxedReplacer;
class LP_TRANSFORMATIONS_API MarkupOptimizations;
class LP_TRANSFORMATIONS_API LowPrecision;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

class ov::pass::low_precision::MarkupOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkupOptimizations", "0");
    MarkupOptimizations(
        const std::vector<PrecisionsRestriction>& precisionRestrictions,
        const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions,
        const AttributeParameters& params);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
private:
    const std::vector<PrecisionsRestriction> precisionRestrictions;
    const std::vector<QuantizationGranularityRestriction> quantizationRestrictions;
    const AttributeParameters params;
};

class ov::pass::low_precision::TypeRelaxedReplacer : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TypeRelaxedReplacer", "0");
    TypeRelaxedReplacer();
};

class ov::pass::low_precision::LowPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("LowPrecision", "0");
    LowPrecision(
        const std::vector<PrecisionsRestriction>& precisionRestrictions = {},
        const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions = {},
        const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static bool isFunctionQuantized(const std::shared_ptr<const ov::Model>& model);
    static bool isFQLevelsPresent(const std::shared_ptr<const ov::Model>& model, const std::set<size_t>& levels);

    template <typename T, class... Args>
    std::shared_ptr<T> add_main(Args&&... args) {
        const auto tr = std::make_shared<T>(std::forward<Args>(args)...);
        additional_main_passes.push_back(tr);
        return tr;
    }

protected:
    std::vector<PrecisionsRestriction> precisionRestrictions;
    std::vector<QuantizationGranularityRestriction> quantizationRestrictions;
    // remove
    LayerTransformation::Params params;

    std::vector<std::shared_ptr<MatcherPass>> additional_main_passes;
};
