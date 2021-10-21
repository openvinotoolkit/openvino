// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

//! [op:header]
namespace TemplateExtension {

class Tile : public ov::op::Op {
public:
    OPENVINO_OP("Tile");

    Tile() = default;
    Tile(const ov::Output<ov::Node>& arg, std::vector<int64_t> repeats);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::runtime::TensorVector& outputs, const ov::runtime::TensorVector& inputs) const override;
    bool has_evaluate() const override;

    std::vector<int64_t> get_repeats() const {
        return repeats;
    }

private:
    std::vector<int64_t> repeats;
};
//! [op:header]

}  // namespace TemplateExtension
