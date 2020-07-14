// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines initialize node runtime information pass
 * @file init_node_info.hpp
 */

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

/**
 * @defgroup ie_transformation_api Inference Engine Transformation API
 * @brief Defines Inference Engine Transformations API which is used to transform ngraph::Function
 *
 * @{
 * @defgroup ie_runtime_attr_api Runtime information
 * @brief A mechanism of runtime information extension
 *
 * @defgroup ie_transformation_common_api Common optimization passes
 * @brief A set of common optimization passes
 *
 * @defgroup ie_transformation_to_opset2_api Conversion from opset3 to opset2
 * @brief A set of conversion downgrade passes from opset3 to opset2
 *
 * @defgroup ie_transformation_to_opset1_api Conversion from opset2 to opset1
 * @brief A set of conversion downgrade passes from opset2 to opset1
 * @}
 */

/**
 * @brief ngraph namespace
 */
namespace ngraph {

/**
 * @brief ngraph::pass namespace
 */
namespace pass {

class TRANSFORMATIONS_API InitNodeInfo;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief InitNodeInfo transformation helps to set runtime info attributes in a single place.
 * 
 * Every runtime info attribute that needs to be initialized should be registered
 * in run_on_function method. Also do not forget to override init methods for registered
 * attribute.
 * This transformations should be called first in transformation pipeline. If attribute was
 * already set initialization will be skipped for this node.
 */
class ngraph::pass::InitNodeInfo: public ngraph::pass::FunctionPass {
public:
	/**
     * Constructor
     */
    InitNodeInfo() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
