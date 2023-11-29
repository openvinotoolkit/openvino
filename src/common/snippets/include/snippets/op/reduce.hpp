// Copyright (C) 2018-2023 Intel Corporation
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
 * @brief
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

protected:
    size_t m_axis;
};

/**
 * @interface ReduceSum
 * @brief
 * @ingroup snippets
 */
class ReduceSum : public ReduceBase {
public:
    OPENVINO_OP("ReduceSum", "SnippetsOpset", ReduceBase);
    ReduceSum(const Output<Node>& x, size_t axis) : ReduceBase(x, axis) {}
    ReduceSum() = default;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    static std::set<ov::element::TypeVector> get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
        return {{ov::element::f32}};
    }
};

/**
 * @interface ReduceMax
 * @brief
 * @ingroup snippets
 */
class ReduceMax : public ReduceBase {
public:
    OPENVINO_OP("ReduceMax", "SnippetsOpset", ReduceBase);
    ReduceMax(const Output<Node>& x, size_t axis) : ReduceBase(x, axis) {}
    ReduceMax() = default;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    static std::set<ov::element::TypeVector> get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
        return {{ov::element::f32}};
    }
};

} // namespace op
} // namespace snippets
} // namespace ov
