// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/decompose_multiply_add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/type_relaxed.hpp"
#include <ngraph/opsets/opset1.hpp>


namespace ngraph {
namespace pass {
namespace low_precision {

void DecomposeMultiplyAddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<ngraph::op::MultiplyAdd>(pass, context);
}

void DecomposeMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    decomposeMultiplyAdd(as_type_ptr<ngraph::op::MultiplyAdd>(m.get_match_root()));
}


}// namespace low_precision
}// namespace pass
}// namespace ngraph