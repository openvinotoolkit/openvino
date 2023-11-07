// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_loss.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeCTCLoss(const ov::Output<Node>& logitsNode,
                                  std::vector<int>& logitsLength,
                                  std::vector<std::vector<int>>& labels,
                                  std::vector<int>& labelsLength,
                                  int blankIndex,
                                  const element::Type& fType,
                                  const element::Type& iType,
                                  const bool preprocessCollapseRepeated,
                                  const bool ctcMergeRepeated,
                                  const bool unique) {
    auto logitsShape = logitsNode.get_shape();
    size_t N = logitsShape[0];
    size_t T = logitsShape[1];

    std::vector<int> labelsOneD(N * T);
    for (int i = 0; i < labels.size(); i++)
        std::copy(labels[i].begin(), labels[i].end(), labelsOneD.data() + i * T);

    auto logitsLengthNode = makeConstant(iType, {N}, logitsLength);
    auto labelsNode = makeConstant(iType, {N, T}, labelsOneD);
    auto labelsLengthNode = makeConstant(iType, {N}, labelsLength);
    auto blankIndexNode = makeConstant<int>(iType, {}, {blankIndex});

    auto ctcLossNode = std::make_shared<ov::op::v4::CTCLoss>(logitsNode,
                                                             logitsLengthNode,
                                                             labelsNode,
                                                             labelsLengthNode,
                                                             blankIndexNode,
                                                             preprocessCollapseRepeated,
                                                             ctcMergeRepeated,
                                                             unique);

    return ctcLossNode;
}

}  // namespace builder
}  // namespace ngraph
