// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeMVN(const ngraph::Output<Node> &in,
                                      bool acrossChannels,
                                      bool normalizeVariance,
                                      double eps) {
    auto mvnNode = std::make_shared<ngraph::op::MVN>(in, acrossChannels, normalizeVariance, eps);

    // Ngraph MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid behavior
    ngraph::AxisSet axes;
    const size_t startAxis = acrossChannels ? 1 : 2;
    const size_t numOfDims = in.get_partial_shape().size();
    for (size_t i = startAxis; i < numOfDims; i++)
        axes.insert(i);
    mvnNode->set_reduction_axes(axes);

    return mvnNode;
}

std::shared_ptr<ngraph::Node> makeMVN(const ngraph::Output<Node> &in,
                                      const ngraph::AxisSet &axes,
                                      bool normalizeVariance,
                                      double eps) {
    auto mvnNode = std::make_shared<ngraph::op::MVN>(in, axes, normalizeVariance, eps);

    return mvnNode;
}

std::shared_ptr<Node> makeMVN6(const Output<Node>& in,
                               const Output<Node>& axesNode,
                               bool normalizeVariance,
                               float eps,
                               std::string& epsMode) {
    op::MVNEpsMode nEpsMode = op::MVNEpsMode::INSIDE_SQRT;
    if (epsMode == "outside_sqrt")
        nEpsMode = op::MVNEpsMode::OUTSIDE_SQRT;
    auto mvnNode = std::make_shared<op::v6::MVN>(in, axesNode, normalizeVariance, eps, nEpsMode);

    return mvnNode;
}

}  // namespace builder
}  // namespace ngraph
