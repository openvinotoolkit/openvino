// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/op_extension.hpp"
#include "openvino/op/op.hpp"

class MockOp : public ov::op::Op {
public:
    OPENVINO_OP("MockOp", "custom_opset");
    MockOp() = default;

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return nullptr;
    }
};

namespace {
auto shared_mock_op_extension = std::make_shared<ov::OpExtension<MockOp>>();
}

OPENVINO_EXTENSION_C_API int get_mock_extension_counter();

int get_mock_extension_counter() {
    const int basic_internal_counter = 1;
    return shared_mock_op_extension.use_count() - basic_internal_counter;
}

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    shared_mock_op_extension,
}));
