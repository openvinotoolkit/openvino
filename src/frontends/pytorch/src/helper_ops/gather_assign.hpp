// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "helper_ops/internal_op.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/op/op.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class GatherAssign : public InternalReverseOperation {
public:
    OPENVINO_OP("GatherAssign", "internal", InternalReverseOperation);

    GatherAssign(const Output<Node>& data,
                 const Output<Node>& updates,
                 const Output<Node>& indices,
                 const Output<Node>& axis)
        : InternalReverseOperation({data, updates, indices, axis}) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data = input_value(0);
        set_output_type(0, data.get_element_type(), data.get_partial_shape());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<GatherAssign>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    }
};
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
