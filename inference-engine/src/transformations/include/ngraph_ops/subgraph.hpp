// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>
#include "ngraph/function.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {


// \brief An operation that is implemented by a function.
class TRANSFORMATIONS_API Subgraph : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Subgraph", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Subgraph(const OutputVector& args, std::shared_ptr<Function> body);

    Subgraph(const NodeVector& args, std::shared_ptr<Function> body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    std::shared_ptr<Function> get_body() const {
        return m_body;
    }

    /// Set a new body for the op; body needs to satisfy requirements on inputs/outputs
    void set_body(std::shared_ptr<Function> body);

private:
    std::shared_ptr<Function> m_body;
};

}  // namespace op
}  // namespace ngraph
