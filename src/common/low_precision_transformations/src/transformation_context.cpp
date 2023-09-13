// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transformation_context.hpp"

namespace ov {
namespace pass {
namespace low_precision {

TransformationContext::TransformationContext() : model(nullptr) {}

TransformationContext::TransformationContext(std::shared_ptr<Model> model) : model(model) {
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
