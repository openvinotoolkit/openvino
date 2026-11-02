// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief Histc operation computes the histogram of all input elements using equal-width bins.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Histc : public Op {
public:
    OPENVINO_OP("Histc", "opset17");

    /// \brief Output element type: SAME_AS_INPUT keeps the input type, I32 always produces i32.
    enum class OutputType { SAME_AS_INPUT, I32 };

    Histc() = default;

    /// \brief Constructs a Histc operation.
    ///
    /// \param data         Input tensor of numeric type and arbitrary shape.
    /// \param bins         Number of equal-width bins.
    /// \param min          Lower end of the range (inclusive). If equal to max, the range is taken from the data.
    /// \param max          Upper end of the range (inclusive). If equal to min, the range is taken from the data.
    /// \param output_type  Selects the output element type.
    Histc(const Output<Node>& data,
          int64_t bins = 100,
          double min = 0.0,
          double max = 0.0,
          OutputType output_type = OutputType::SAME_AS_INPUT);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_bins() const {
        return m_bins;
    }
    void set_bins(int64_t bins) {
        m_bins = bins;
    }
    double get_min() const {
        return m_min;
    }
    void set_min(double min) {
        m_min = min;
    }
    double get_max() const {
        return m_max;
    }
    void set_max(double max) {
        m_max = max;
    }
    OutputType get_output_type() const {
        return m_output_type;
    }
    void set_output_type(OutputType output_type) {
        m_output_type = output_type;
    }
    using Node::set_output_type;

private:
    int64_t m_bins{100};
    double m_min{0.0};
    double m_max{0.0};
    OutputType m_output_type{OutputType::SAME_AS_INPUT};
};
}  // namespace ov::op::v17

namespace ov {
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const op::v17::Histc::OutputType& type);

template <>
class OPENVINO_API AttributeAdapter<op::v17::Histc::OutputType>
    : public EnumAttributeAdapterBase<op::v17::Histc::OutputType> {
public:
    AttributeAdapter(op::v17::Histc::OutputType& value) : EnumAttributeAdapterBase<op::v17::Histc::OutputType>(value) {}
    ~AttributeAdapter() override;

    OPENVINO_RTTI("AttributeAdapter<ov::op::v17::Histc::OutputType>");
};
}  // namespace ov
