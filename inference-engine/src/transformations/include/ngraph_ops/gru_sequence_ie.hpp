// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <transformations_visibility.hpp>

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
class TRANSFORMATIONS_API GRUSequenceIE : public Op {
public:
GRUSequenceIE(const Output<Node> &X,
              const Output<Node> &H_t,
              const Output<Node>& sequence_lengths,
              const Output<Node> &WR,
              const Output<Node> &B,
              size_t hidden_size,
              op::RecurrentSequenceDirection direction,
              const std::vector<std::string>& activations,
              const std::vector<float>& activations_alpha,
              const std::vector<float>& activations_beta,
              float clip);

static constexpr NodeTypeInfo type_info{"GRUSequenceIE", 1};
const NodeTypeInfo& get_type_info() const override { return type_info; }

GRUSequenceIE() = delete;

std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
void validate_and_infer_types() override;

std::size_t get_hidden_size() { return m_hidden_size; }
const std::vector<std::string>& get_activations() { return m_activations; }
const std::vector<float>& get_activations_alpha() { return m_activations_alpha; }
const std::vector<float>& get_activations_beta() { return m_activations_beta; }
float get_clip() {return m_clip;}

protected:
int64_t m_hidden_size{};

const std::vector<std::string> m_activations;
const std::vector<float> m_activations_alpha;
const std::vector<float>  m_activations_beta;
op::RecurrentSequenceDirection m_direction;
float m_clip;
};

}  // namespace op
}  // namespace ngraph
