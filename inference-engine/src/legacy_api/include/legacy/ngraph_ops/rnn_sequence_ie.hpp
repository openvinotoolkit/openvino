// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_api.h>

#include "ngraph/opsets/opset4.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
class INFERENCE_ENGINE_API_CLASS(RNNSequenceIE) : public ngraph::op::util::RNNCellBase {
public:
    NGRAPH_RTTI_DECLARATION;

    RNNSequenceIE(const Output <Node> &X,
                  const Output <Node> &H_t,
                  const Output <Node> &seq_lengths,
                  const Output <Node> &WR,
                  const Output <Node> &B,
                  size_t hidden_size,
                  op::RecurrentSequenceDirection direction,
                  const std::vector<std::string> &activations,
                  const std::vector<float> &activations_alpha,
                  const std::vector<float> &activations_beta,
                  float clip,
                  int64_t seq_axis = 1);

    RNNSequenceIE() = delete;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override;

    void validate_and_infer_types() override;

    std::size_t get_hidden_size() { return m_hidden_size; }

    const std::vector<std::string> &get_activations() { return m_activations; }

    const std::vector<float> &get_activations_alpha() { return m_activations_alpha; }

    const std::vector<float> &get_activations_beta() { return m_activations_beta; }

    float get_clip() { return m_clip; }

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    op::RecurrentSequenceDirection m_direction;
    int64_t m_seq_axis;
};
}  // namespace op
}  // namespace ngraph
