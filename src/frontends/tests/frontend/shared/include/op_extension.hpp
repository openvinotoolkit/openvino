// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/manager.hpp"

struct OpExtensionFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
    std::vector<std::shared_ptr<ov::Extension>> m_extensions;
};

class Relu : public ov::op::Op {
public:
    OPENVINO_OP("Relu", "frontend_test");

    Relu() = default;
    Relu(const ov::Output<ov::Node>& arg) : ov::op::Op({arg}) {
        validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        ov::element::Type arg_type = get_input_element_type(0);
        ov::PartialShape arg_shape = get_input_partial_shape(0);
        set_output_type(0, arg_type, arg_shape);
    }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<Relu>(new_args.at(0));
    }
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        return false;
    }
    bool has_evaluate() const override {
        return false;
    }
};

class FrontEndOpExtensionTest : public ::testing::TestWithParam<OpExtensionFEParam> {
public:
    OpExtensionFEParam m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<OpExtensionFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
