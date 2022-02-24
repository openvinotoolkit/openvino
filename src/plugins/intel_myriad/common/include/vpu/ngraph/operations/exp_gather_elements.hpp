// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph { namespace vpu { namespace op {

//
// ExpGatherElements is an extended version of GatherElements-6.
// Besides data, indices and axis, it has an additional input - lookupIndices and an additional attribute - lookupAxis
// which are working in the same way as in Gather-1, so the output data is calculating by the following formula:
// output[:, ..., i, j, k, ..., m, ...] = input[:, ..., lookupIndices[i, j, k], ..., indices[:, ..., i, j, k, ..., m, ...], ...]
// where m is axis and i is lookupAxis.
// Output shape is the same as indices shape
//

class ExpGatherElements : public ngraph::op::Op {
public:
    OPENVINO_OP("ExpGatherElements", "VPUOpset");

    ExpGatherElements(const Output<Node>& data,
                      const Output<Node>& indices,
                      const Output<Node>& lookupIndices,
                      const int64_t axis,
                      const int64_t lookupAxis);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node>
    clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() const { return m_axis; }
    int64_t get_lookup_axis() const { return m_lookup_axis; }
private:
    int64_t m_axis;
    int64_t m_lookup_axis;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
