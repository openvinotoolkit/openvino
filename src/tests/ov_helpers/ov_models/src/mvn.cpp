// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn.hpp"

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeMVN(const ov::Output<Node>& in, bool acrossChannels, bool normalizeVariance, double eps) {
    auto mvnNode = std::make_shared<ov::op::v0::MVN>(in, acrossChannels, normalizeVariance, eps);

    // OpenVINO MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid behavior
    ov::AxisSet axes;
    const size_t startAxis = acrossChannels ? 1 : 2;
    const size_t numOfDims = in.get_partial_shape().size();
    for (size_t i = startAxis; i < numOfDims; i++)
        axes.insert(i);
    mvnNode->set_reduction_axes(axes);

    return mvnNode;
}

std::shared_ptr<ov::Node> makeMVN(const ov::Output<Node>& in,
                                  const ov::AxisSet& axes,
                                  bool normalizeVariance,
                                  double eps) {
    auto mvnNode = std::make_shared<ov::op::v0::MVN>(in, axes, normalizeVariance, eps);

    return mvnNode;
}

std::shared_ptr<Node> makeMVN6(const Output<Node>& in,
                               const Output<Node>& axesNode,
                               bool normalizeVariance,
                               float eps,
                               std::string& epsMode) {
    ov::op::MVNEpsMode nEpsMode = ov::op::MVNEpsMode::INSIDE_SQRT;
    if (epsMode == "outside_sqrt")
        nEpsMode = ov::op::MVNEpsMode::OUTSIDE_SQRT;
    auto mvnNode = std::make_shared<ov::op::v6::MVN>(in, axesNode, normalizeVariance, eps, nEpsMode);

    return mvnNode;
}

}  // namespace builder
}  // namespace ngraph
