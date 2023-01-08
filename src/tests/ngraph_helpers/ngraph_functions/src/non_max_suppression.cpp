// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

template <typename NmsOperation>
std::shared_ptr<ngraph::Node> makeNms(const ngraph::Output<Node>& boxes,
                                      const ngraph::Output<Node>& scores,
                                      const element::Type& maxBoxesPrec,
                                      const element::Type& thrPrec,
                                      const int32_t& maxOutBoxesPerClass,
                                      const float& iouThr,
                                      const float& scoreThr,
                                      const float& softNmsSigma,
                                      const bool& isCenter,
                                      const bool& sortResDescend,
                                      const ngraph::element::Type& outType) {
    auto maxOutBoxesPerClassNode =
        makeConstant(maxBoxesPrec, ngraph::Shape{}, std::vector<int32_t>{maxOutBoxesPerClass})->output(0);
    auto iouThrNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{iouThr})->output(0);
    auto scoreThrNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{scoreThr})->output(0);
    auto softNmsSigmaNode = makeConstant(thrPrec, ngraph::Shape{}, std::vector<float>{softNmsSigma})->output(0);

    typename NmsOperation::BoxEncodingType boxEncodingType =
        isCenter ? NmsOperation::BoxEncodingType::CENTER : NmsOperation::BoxEncodingType::CORNER;

    return std::make_shared<NmsOperation>(boxes,
                                          scores,
                                          maxOutBoxesPerClassNode,
                                          iouThrNode,
                                          scoreThrNode,
                                          softNmsSigmaNode,
                                          boxEncodingType,
                                          sortResDescend,
                                          outType);
}

std::shared_ptr<ngraph::Node> makeNms(const ngraph::Output<Node>& boxes,
                                      const ngraph::Output<Node>& scores,
                                      const element::Type& maxBoxesPrec,
                                      const element::Type& thrPrec,
                                      const int32_t& maxOutBoxesPerClass,
                                      const float& iouThr,
                                      const float& scoreThr,
                                      const float& softNmsSigma,
                                      const bool isCenter,
                                      const bool& sortResDescend,
                                      const ngraph::element::Type& outType,
                                      const NmsVersion nmsVersion) {
    switch (nmsVersion) {
    case NmsVersion::NmsVersion5:
        return makeNms<opset5::NonMaxSuppression>(boxes,
                                                  scores,
                                                  maxBoxesPrec,
                                                  thrPrec,
                                                  maxOutBoxesPerClass,
                                                  iouThr,
                                                  scoreThr,
                                                  softNmsSigma,
                                                  isCenter,
                                                  sortResDescend,
                                                  outType);
    default:
        return makeNms<opset9::NonMaxSuppression>(boxes,
                                                  scores,
                                                  maxBoxesPrec,
                                                  thrPrec,
                                                  maxOutBoxesPerClass,
                                                  iouThr,
                                                  scoreThr,
                                                  softNmsSigma,
                                                  isCenter,
                                                  sortResDescend,
                                                  outType);
    }
}

}  // namespace builder
}  // namespace ngraph
