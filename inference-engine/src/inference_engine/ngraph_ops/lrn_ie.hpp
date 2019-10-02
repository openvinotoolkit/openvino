//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class LRN_IE : public Op {
public:
    LRN_IE() = default;

    LRN_IE(const Output<Node>& arg,
        double alpha,
        double beta,
        double bias,
        size_t size,
        std::string region);

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
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
