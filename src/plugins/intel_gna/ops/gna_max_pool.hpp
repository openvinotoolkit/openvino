// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "ngraph/node.hpp"
#include "openvino/op/util/max_pool_base.hpp"

namespace GNAPluginNS {
namespace Op {
namespace v1 {
/// \brief Batched max pooling operation.
class GNAMaxPool : public ov::op::util::MaxPoolBase {
public:
    NGRAPH_RTTI_DECLARATION;

    /// \brief Constructs a batched max pooling operation.
    GNAMaxPool() = default;

    /// \brief Constructs a batched max pooling operation.
    ///
    /// \param arg The node producing the input data batch tensor.
    /// \param strides The strides.
    /// \param pads_begin The beginning of padding shape.
    /// \param pads_end The end of padding shape.
    /// \param kernel The kernel shape.
    /// \param rounding_type Whether to use ceiling or floor rounding type while
    /// computing output shape.
    /// \param auto_pad The pad type for automatically computing padding sizes.
    GNAMaxPool(const ngraph::Output<ngraph::Node>& arg,
            const ngraph::Strides& strides,
            const ngraph::Shape& pads_begin,
            const ngraph::Shape& pads_end,
            const ngraph::Shape& kernel,
            const ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR,
            const ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    /// \return The default value for MaxPool.
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<Node> get_default_value() const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

private:
    bool evaluate_maxpool(const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) const;
};
}  // namespace v1
} // namespace Op
} // namespace GNAPluginNS
