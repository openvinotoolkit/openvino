// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeNonMaxSuppression : public ngraph::op::NonMaxSuppressionIE3 {
public:
    OPENVINO_OP("StaticShapeNonMaxSuppression", "VPUOpset", ngraph::op::NonMaxSuppressionIE3);

    explicit StaticShapeNonMaxSuppression(const ngraph::opset5::NonMaxSuppression& nms);

    StaticShapeNonMaxSuppression(const Output<Node>& boxes,
                                 const Output<Node>& scores,
                                 const Output<Node>& maxOutputBoxesPerClass,
                                 const Output<Node>& iouThreshold,
                                 const Output<Node>& scoreThreshold,
                                 const Output<Node>& softNmsSigma,
                                 int centerPointBox = 0,
                                 bool sortResultDescending = true,
                                 const ngraph::element::Type& outputType = ngraph::element::i64);

    void validate_and_infer_types() override;
    void set_output_type(const ngraph::element::Type& output_type);
    using Node::set_output_type;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
