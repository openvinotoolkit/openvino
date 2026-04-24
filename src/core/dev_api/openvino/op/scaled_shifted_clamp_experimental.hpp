// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace experimental {

/// \note Experimental op. API is subject to change or removal.
///
/// \brief Computes y = clamp(x * scale + bias, lo, hi) elementwise.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScaledShiftedClamp : public ov::op::Op {
public:
    OPENVINO_OP("ScaledShiftedClampExperimental");

    ScaledShiftedClamp() = default;
    ScaledShiftedClamp(const Output<Node>& data, double scale, double bias, double lo, double hi);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    double get_scale() const {
        return m_scale;
    }
    double get_bias() const {
        return m_bias;
    }
    double get_lo() const {
        return m_lo;
    }
    double get_hi() const {
        return m_hi;
    }

private:
    double m_scale{1.0};
    double m_bias{0.0};
    double m_lo{std::numeric_limits<double>::lowest()};
    double m_hi{std::numeric_limits<double>::max()};
};

}  // namespace experimental
}  // namespace op
}  // namespace ov
