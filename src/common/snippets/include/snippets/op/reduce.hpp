// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface ReduceBase
 * @brief Base class for reduce operations.
 * @param m_axis reduce axis.
 * @ingroup snippets
 */
class ReduceBase : public ov::op::Op {
public:
    OPENVINO_OP("ReduceBase", "SnippetsOpset");

    ReduceBase(const Output<Node>& x, size_t axis);
    ReduceBase() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    size_t get_axis() const { return m_axis; }
    static void compute_and_set_reduce_subtensors(const std::shared_ptr<ReduceBase>& reduce);

protected:
    size_t m_axis = 0;
};

class ReduceSum : public ReduceBase {
public:
    OPENVINO_OP("ReduceSum", "SnippetsOpset", ReduceBase);
    ReduceSum(const Output<Node>& x, size_t axis) : ReduceBase(x, axis) {}
    ReduceSum() = default;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

class ReduceMax : public ReduceBase {
public:
    OPENVINO_OP("ReduceMax", "SnippetsOpset", ReduceBase);
    ReduceMax(const Output<Node>& x, size_t axis) : ReduceBase(x, axis) {}
    ReduceMax() = default;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

} // namespace op
} // namespace snippets
} // namespace ov
