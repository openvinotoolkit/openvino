// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/util.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/strides.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API StrideOptimization;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief StrideOptimization transformation is a FunctionPass
 * that propagates stride (greater than 1) from Convolution
 * up through the graph (namely Relu, Maximum, Mul, Add and Conv operators)
 */
class ngraph::pass::StrideOptimization: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
private:
    bool handle_node(std::shared_ptr<ngraph::Node>& node);
    bool conv_stride_propagation(std::shared_ptr<ngraph::Node>& conv);
    bool simple_stride_propagation(std::shared_ptr<ngraph::Node>& conv, bool supported);
    std::tuple<std::vector<Strides>, bool> check_next_ops(const std::vector<std::shared_ptr<Node>>& next_ops);
    void insert_pooling(const std::shared_ptr<Node>& first, const std::shared_ptr<Node>& second, const Strides& strides);

    std::map<std::string, Strides> m_strides_map;
};
