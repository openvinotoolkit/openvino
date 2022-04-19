// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

// one place to include all Low Precision Transformations from ngraph::pass::low_precision
#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <low_precision/rt_info/precisions_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>

#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precision_preserved.hpp>
#include <low_precision/propagate_precisions.hpp>
#include <low_precision/align_quantization_intervals.hpp>


#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/common/quantization_granularity_restriction.hpp>
#include <low_precision/common/precisions_restriction.hpp>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/markup_precisions.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API TypeRelaxedReplacer;
class LP_TRANSFORMATIONS_API MarkupOptimizations;
class LP_TRANSFORMATIONS_API LowPrecision;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::MarkupOptimizations : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("MarkupOptimizations", "0");
    MarkupOptimizations(
        const std::vector<PrecisionsRestriction>& precisionRestrictions,
        const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions,
        const AttributeParameters& params);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    const std::vector<PrecisionsRestriction>& precisionRestrictions;
    const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions;
    const AttributeParameters& params;
};

class ngraph::pass::low_precision::TypeRelaxedReplacer : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TypeRelaxedReplacer", "0");
    TypeRelaxedReplacer();
};

class ngraph::pass::low_precision::LowPrecision : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("LowPrecision", "0");
    LowPrecision(
        const std::vector<PrecisionsRestriction>& precisionRestrictions = {},
        const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions = {},
        const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

    static bool isFunctionQuantized(const std::shared_ptr<const ngraph::Function>& function);
    static bool isFQLevelsPresent(const std::shared_ptr<const ngraph::Function>& function, const std::set<size_t>& levels);

protected:
    std::vector<PrecisionsRestriction> precisionRestrictions;
    std::vector<QuantizationGranularityRestriction> quantizationRestrictions;
    // remove
    LayerTransformation::Params params;
};
