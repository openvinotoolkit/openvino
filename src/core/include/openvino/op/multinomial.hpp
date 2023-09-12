// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Multinomial operation over the input tensor.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Multinomial : public Op {
public:
    OPENVINO_OP("Multinomial", "opset13");
    Multinomial();
    /// \param input The input tensor to be normalized
    /// \param num_samples The tensor containing samples count. Defines output shape of the network
    /// \param output_type The data type of the output
    /// \param with_replacement Determines sampling with replacement
    /// \param log_probs Determines whether the input contains probabilities or log probabilities
    /// \param global_seed The seed used in ----
    /// \param op_seed The seed used in ----
    Multinomial(const Output<Node>& input,
                const Output<Node>& num_samples,
                const ngraph::element::Type_t output_type,
                const bool with_replacement,
                const bool log_probs,
                const int64_t global_seed,
                const int64_t op_seed);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    ngraph::element::Type_t get_output_type() const {
        return m_output_type;
    }

    void set_output_type(ngraph::element::Type_t output_type) {
        m_output_type = output_type;
    }

    bool get_with_replacement() const {
        return m_with_replacement;
    }

    void set_with_replacement(bool with_replacement) {
        m_with_replacement = with_replacement;
    }

    bool get_log_probs() const {
        return m_log_probs;
    }

    void set_log_probs(bool log_probs) {
        m_log_probs = log_probs;
    }

    int64_t get_global_seed() const {
        return m_global_seed;
    }

    void set_global_seed(int64_t global_seed) {
        m_global_seed = global_seed;
    }

    int64_t get_op_seed() const {
        return m_op_seed;
    }

    void set_op_seed(int64_t op_seed) {
        m_op_seed = op_seed;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    ngraph::element::Type_t m_output_type;
    bool m_with_replacement;
    bool m_log_probs;
    int64_t m_global_seed;
    int64_t m_op_seed;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
