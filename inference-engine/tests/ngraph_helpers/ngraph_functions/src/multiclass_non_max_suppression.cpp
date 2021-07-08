// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeMulticlassNms(
    const ngraph::Output<Node> &boxes, const ngraph::Output<Node> &scores,
    const element::Type &maxBoxesPrec, const element::Type &thrPrec,
    const int32_t &maxOutBoxesPerClass, const float &iouThr,
    const float &scoreThr, const int32_t &backgroundClass,
    const int32_t &keepTopK, const ngraph::element::Type &outType,
    const ngraph::op::util::NmsBase::SortResultType
        sortResultType,
    const bool &sortResCB, const float &nmsEta, const bool &normalized) {
  // auto maxOutBoxesPerClassNode = makeConstant(maxBoxesPrec, ngraph::Shape{},
  // std::vector<int32_t>{maxOutBoxesPerClass})->output(0); auto iouThrNode =
  // makeConstant(thrPrec, ngraph::Shape{},
  // std::vector<float>{iouThr})->output(0); auto scoreThrNode =
  // makeConstant(thrPrec, ngraph::Shape{},
  // std::vector<float>{scoreThr})->output(0);
  // // auto softNmsSigmaNode = makeConstant(thrPrec, ngraph::Shape{},
  // std::vector<float>{softNmsSigma})->output(0); auto backgroundClassNode =
  // makeConstant(maxBoxesPrec, ngraph::Shape{},
  // std::vector<int32_t>{backgroundClass})->output(0); auto keepTopKNode =
  // makeConstant(maxBoxesPrec, ngraph::Shape{},
  // std::vector<int32_t>{keepTopK})->output(0);

  // return std::make_shared<ngraph::op::v5::MulticlassNonMaxSuppression>(boxes,
  // scores, maxOutBoxesPerClassNode, iouThrNode, scoreThrNode,
  // backgroundClassNode,
  //                                                            keepTopKNode,
  //                                                            outType);

  ngraph::op::v8::MulticlassNms::Attributes attrs;
  attrs.sort_result_type = sortResultType;
  attrs.sort_result_across_batch = sortResCB;
  attrs.output_type = outType;
  attrs.iou_threshold = iouThr;
  attrs.score_threshold = scoreThr;
  attrs.nms_top_k = maxOutBoxesPerClass;
  attrs.keep_top_k = keepTopK;
  attrs.background_class = backgroundClass;
  attrs.nms_eta = nmsEta;
  attrs.normalized = normalized;

  return std::make_shared<ngraph::op::v8::MulticlassNms>(boxes, scores, attrs);
}

} // namespace builder
} // namespace ngraph
