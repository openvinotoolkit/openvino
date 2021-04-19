// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines dimension finding and tracking pass
 * @file dimension_tracking.hpp
 */

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

/**
 * @brief ngraph namespace
 */
namespace ngraph {

/**
* @brief ngraph::pass namespace
*/
namespace pass {

class TRANSFORMATIONS_API FindBatch;
class TRANSFORMATIONS_API TrackDimensions;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief FindBatch transformation tracks down dimension starting from all the
 * Parameters of the Function.
 */
class ngraph::pass::FindBatch: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
