// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeNonMaxSuppression : public ngraph::opset5::NonMaxSuppression {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeNonMaxSuppression", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    explicit StaticShapeNonMaxSuppression(const ngraph::opset5::NonMaxSuppression& nms);

    StaticShapeNonMaxSuppression(const Output<Node>& boxes,
                                 const Output<Node>& scores,
                                 const Output<Node>& maxOutputBoxesPerClass,
                                 const Output<Node>& iouThreshold,
                                 const Output<Node>& scoreThreshold,
                                 const Output<Node>& softNmsSigma,
                                 ngraph::op::v5::NonMaxSuppression::BoxEncodingType boxEncodingType = ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
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
