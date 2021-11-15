// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transformation_context.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

TransformationContext::TransformationContext() : function(nullptr) {}

TransformationContext::TransformationContext(std::shared_ptr<Function> function) : function(function) {
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
