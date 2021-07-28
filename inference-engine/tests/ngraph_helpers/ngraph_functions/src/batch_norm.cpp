// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeBatchNormInference(const ngraph::Output<Node>& data,
                                                     double epsilon) {
    auto ngPrc = data.get_element_type();
    size_t C   = data.get_shape().at(1);
    bool random = true;
    std::vector<float> values(C);
    auto gamma = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{C}, values, random, 1.f, 0.f);
    auto beta  = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{C}, values, random, 1.f, 0.f);
    auto mean  = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{C}, values, random, 1.f, 0.f);

    // Fill the vector for variance with positive values
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dis(0.0, 10.0);
    std::generate(values.begin(), values.end(), [&dis, &gen]() { return dis(gen); });
    auto variance = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{C}, values, !random);
    return std::make_shared<ngraph::opset5::BatchNormInference>(data, gamma, beta, mean, variance, epsilon);
}
}  // namespace builder
}  // namespace ngraph
