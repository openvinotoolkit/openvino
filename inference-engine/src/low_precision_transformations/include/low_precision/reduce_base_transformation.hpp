// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
* @brief ReduceBaseTransformation: base class for Reduce*Transformation
* detects dequantization operations in front of the Reduce* layer and
* propagates them through the Reduce* if possible
* 
*/

class TRANSFORMATIONS_API ReduceBaseTransformation : public LayerTransformation {
public:
    ReduceBaseTransformation(const Params& params);
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const override;

protected:
    virtual void changeDequantizationValues(
        const std::shared_ptr<Node>& reduce,
        FakeQuantizeDequantization& dequantization) const;
    virtual bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
