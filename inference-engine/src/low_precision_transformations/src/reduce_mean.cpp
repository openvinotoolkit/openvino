// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_mean.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceMeanTransformation::ReduceMeanTransformation(const Params& params) : ReduceBaseTransformation(params) {}

void ReduceMeanTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::ReduceMean>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool ReduceMeanTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    return is_type<opset1::ReduceMean>(reduce) ? ReduceBaseTransformation::canBeTransformed(context, reduce) : false;
}

bool ReduceMeanTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return false;
}

bool ReduceMeanTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
