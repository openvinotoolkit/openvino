// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeNms(const ngraph::Output<Node> &boxes,
                                      const ngraph::Output<Node> &scores,
                                      const ngraph::Output<Node> &maxBoxes,
                                      const element::Type& thrPrec,
                                      const float &iouThr,
                                      const float &scoreThr,
                                      const float &softNmsSigma,
                                      const ngraph::op::v5::NonMaxSuppression::BoxEncodingType &boxEncoding,
                                      const bool &sortResDescend,
                                      const ngraph::element::Type& outType) {
    auto iouThrNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{iouThr})->output(0);
    auto scoreThrNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{scoreThr})->output(0);
    auto softNmsSigmaNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{softNmsSigma})->output(0);

    return std::make_shared<ngraph::op::v5::NonMaxSuppression>(boxes, scores, maxBoxes, iouThrNode, scoreThrNode, softNmsSigmaNode,
                                                               boxEncoding, sortResDescend, outType);
}

}  // namespace builder
}  // namespace ngraph
