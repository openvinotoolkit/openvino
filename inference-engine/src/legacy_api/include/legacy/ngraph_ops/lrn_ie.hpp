// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(LRN_IE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"LRN_IE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    LRN_IE() = default;

    LRN_IE(const Output<Node>& arg,
        double alpha,
        double beta,
        double bias,
        size_t size,
        std::string region);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    double get_alpha() const { return m_alpha; }
    void set_alpha(double alpha) { m_alpha = alpha; }
    double get_beta() const { return m_beta; }
    void set_beta(double beta) { m_beta = beta; }
    double get_bias() const { return m_bias; }
    void set_bias(double bias) { m_bias = bias; }
    size_t get_nsize() const { return m_size; }
    void set_nsize(size_t size) { m_size = size; }
    std::string get_region() const { return m_region; }
    void set_region(std::string region) { m_region = region; }

protected:
    double m_alpha;
    double m_beta;
    double m_bias;
    size_t m_size;
    std::string m_region;
};

}  // namespace op
}  // namespace ngraph
