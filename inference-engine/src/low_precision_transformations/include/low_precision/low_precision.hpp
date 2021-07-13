// Copyright (C) 2021 Intel Corporation
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
#include <low_precision/common/operation_per_tensor_quantization_restriction.hpp>
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

class LP_TRANSFORMATIONS_API ngraph::pass::low_precision::MarkupOptimizations : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupOptimizations(
        const std::vector<OperationPrecisionRestriction>& precisionRestrictions,
        const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions);
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
private:
    const std::vector<OperationPrecisionRestriction>& precisionRestrictions;
    const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions;
};

class LP_TRANSFORMATIONS_API ngraph::pass::low_precision::TypeRelaxedReplacer : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TypeRelaxedReplacer();
};

class LP_TRANSFORMATIONS_API ngraph::pass::low_precision::LowPrecision : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowPrecision(
        const std::vector<OperationPrecisionRestriction>& precisionRestrictions = {},
        const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions = {},
        const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    static bool isFunctionQuantized(const std::shared_ptr<const ngraph::Function>& function);

protected:
    std::vector<OperationPrecisionRestriction> precisionRestrictions;
    std::vector<OperationPerTensorQuantizationRestriction> quantizationRestrictions;
    // remove
    LayerTransformation::Params params;
};
