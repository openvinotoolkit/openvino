// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ApplyTransformationsToTIBody;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Applies transformations from given pass::Manager for each TensorIterator body
 */

class ngraph::pass::ApplyTransformationsToTIBody: public ngraph::pass::MatcherPass {
public:
    explicit ApplyTransformationsToTIBody(ngraph::pass::Manager & manager);
};
