// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API NormalizeIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"NormalizeIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

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
