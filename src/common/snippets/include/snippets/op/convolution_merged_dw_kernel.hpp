// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"
//#include "tile.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ngraph {
namespace snippets {
namespace op {

class ConvolutionMergedDwKernel : public ngraph::op::Op {
public:
    OPENVINO_OP("ConvolutionMergedDwKernel", "SnippetsOpset");

    ConvolutionMergedDwKernel(
            const std::vector<Output<Node>>& data_batch,
            const Output<Node>& filters,
            const Output<Node>& biases,
            const Strides& strides,
            const ov::CoordinateDiff& pads_begin,
            const ov::CoordinateDiff& pads_end,
            const Strides& dilations,
            const ov::op::PadType& auto_pad,
            const size_t outputs_size);

    bool visit_attributes(AttributeVisitor& visitor) override { return true; };
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    ov::CoordinateDiff get_pads_begin() const;
    ov::CoordinateDiff get_pads_end() const;

    size_t outputs_size;

private:
    const Strides strides;
    const ov::CoordinateDiff pads_begin;
    const ov::CoordinateDiff pads_end;
    const Strides dilations;
    const ov::op::PadType& auto_pad;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
