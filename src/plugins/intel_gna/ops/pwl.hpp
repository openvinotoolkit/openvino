// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gna {
namespace op {
/// \brief PWL activation function
class Pwl : public ov::op::Op {
public:
    OPENVINO_OP("Pwl", "intel_gna", ov::op::Op);

    Pwl() = default;
    /// \brief Constructs a Pwl node.
    ///
    /// \param data  - The input data tensor.
    /// \param m     - The list of the slopes of segment.
    /// \param b     - The list of the y-intercepts of segment.
    /// \param knots - The list of x-coordinates of segment endpoints (segments number + 1).
    Pwl(const ngraph::Output<ngraph::Node>& data,
        const ngraph::Output<ngraph::Node>& m,
        const ngraph::Output<ngraph::Node>& b,
        const ngraph::Output<ngraph::Node>& knots);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const ov::EvaluationContext& evaluation_context) const override;
    bool has_evaluate() const override;
    void set_base_node(const std::shared_ptr<ngraph::Node>& base_node);
    std::shared_ptr<ngraph::Node> get_base_node();

private:
    template<typename T1>
    bool evaluate_pwl(const std::tuple<>&, ov::TensorVector&, const ov::TensorVector&) const {
        return false;
    }

    template<typename ...Types2>
    bool evaluate_pwl(const std::tuple<>&, const std::tuple<Types2...>&, ov::TensorVector&, const ov::TensorVector&) const {
        return false;
    }

    template<typename T1, typename ...Types1, typename ...Types2>
    bool evaluate_pwl(const std::tuple<T1, Types1...>&,
                      const std::tuple<Types2...>& types2,
                      ov::TensorVector& outputs,
                      const ov::TensorVector& inputs) const {
        if (evaluate_pwl<T1>(types2, outputs, inputs)) {
            return true;
        }

        return evaluate_pwl(std::tuple<Types1...>(), types2, outputs, inputs);
    }

    template<typename T1, typename T2, typename ...Types2>
    bool evaluate_pwl(const std::tuple<T2, Types2...>&,
                      ov::TensorVector& outputs,
                      const ov::TensorVector& inputs) const {
        return inputs[1].get_element_type() == T1::value &&
               inputs[0].get_element_type() == T2::value &&
               evaluate<T1, T2>(outputs, inputs) ||
               evaluate_pwl<T1>(std::tuple<Types2...>(), outputs, inputs);
    }

    template <typename T1, typename T2>
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const;

    std::shared_ptr<ngraph::Node> m_base_node;
}; // class Pwl
} // namespace op
} // namespace intel_gna
} // namespace ov
