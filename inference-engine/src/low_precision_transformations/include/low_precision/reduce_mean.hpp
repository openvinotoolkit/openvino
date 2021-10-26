// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/reduce_base_transformation.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ReduceMeanTransformation : public ReduceBaseTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ReduceMeanTransformation(const Params& params = Params());
    bool isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const override;

protected:
    bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
