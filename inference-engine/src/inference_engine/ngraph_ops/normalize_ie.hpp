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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class NormalizeIE : public Op {
public:
    NormalizeIE() = default;

    NormalizeIE(const Output<Node>& data,
                const Output<Node>& weights,
                float eps,
                bool across_spatial,
                bool channel_shared);

    float get_eps() const { return m_eps; }
    bool get_channel_shared() const  { return m_channel_shared;}
    bool get_across_spatial() const  { return m_across_spatial;}

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

protected:
    float m_eps;
    bool m_across_spatial;
    bool m_channel_shared;
};

}  // namespace op
}  // namespace ngraph
