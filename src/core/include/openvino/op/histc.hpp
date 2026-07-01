// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v17 {
/// \brief Computes a histogram of the flattened floating-point input tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Histc : public Op {
public:
    OPENVINO_OP("Histc", "opset17", op::Op);

    Histc() = default;

    /// \brief Constructs a Histc operation.
    ///
    /// \param data     Floating-point input tensor
    /// \param bins     Number of histogram bins; defaults to 100
    /// \param min_val  Lower range boundary; defaults to 0.0
    /// \param max_val  Upper range boundary; defaults to 0.0
    Histc(const Output<Node>& data, int64_t bins = 100, double min_val = 0.0, double max_val = 0.0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

    int64_t get_bins() const {
        return m_bins;
    }
    void set_bins(int64_t bins) {
        m_bins = bins;
    }

    double get_min_val() const {
        return m_min_val;
    }
    void set_min_val(double min_val) {
        m_min_val = min_val;
    }

    double get_max_val() const {
        return m_max_val;
    }
    void set_max_val(double max_val) {
        m_max_val = max_val;
    }

private:
    int64_t m_bins = 100;
    double m_min_val = 0.0;
    double m_max_val = 0.0;
};
}  // namespace v17
}  // namespace op
}  // namespace ov
