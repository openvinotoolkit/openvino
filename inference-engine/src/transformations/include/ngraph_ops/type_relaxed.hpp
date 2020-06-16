// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <algorithm>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {


class TypeRelaxedBase {
public:
    TypeRelaxedBase() {}
    virtual ~TypeRelaxedBase() {};

    TypeRelaxedBase(element::Type use_type) :m_output_data_type(use_type) {
    }

    /// \return Data type that will be set for all outputs and overriden for all inputs
    const element::Type& get_overriden_output_type() const { return m_output_data_type; }
    void set_overriden_output_type(const element::Type& element_type) { m_output_data_type = element_type; }

protected:
    // Data types that are used for parent shape/type infer function
    std::vector<element::Type> m_input_data_types;
    element::Type m_output_data_type = element::f32;
};

template <typename BaseOp>
class AutoReplaceInputTypes {
    std::vector<element::Type> m_input_data_types;
    const BaseOp& m_op;
public:
    AutoReplaceInputTypes(const BaseOp& op, element::Type use_type) : m_op(op) {
        // Remember all input data types and reset them to m_output_data_type.
        for (size_t i = 0; i < op.get_input_size(); ++i) {
            m_input_data_types.push_back(op.get_input_element_type(i));
            op.get_input_tensor(i).set_tensor_type(use_type, op.get_input_partial_shape(i));
        }
    }

    operator const BaseOp& () const {
        return m_op;
    }

    ~AutoReplaceInputTypes() {
        // Restore original input data types
        for (size_t i = 0; i < m_op.get_input_size(); ++i) {
            m_op.get_input_tensor(i).set_tensor_type(m_input_data_types[i], m_op.get_input_partial_shape(i));
        }
    }
};


class AutoReplaceOutputType {
    std::shared_ptr<Node> m_node;
    element::Type orig_type;
public:
    AutoReplaceOutputType(std::shared_ptr<Node> node, element::Type tmp_type) : m_node(node) {
        orig_type = node->get_output_element_type(0);
        m_node->set_output_type(0, tmp_type, m_node->get_output_partial_shape(0));
    }

    operator std::shared_ptr<Node>() const {
        return m_node;
    }

    ~AutoReplaceOutputType() {
        m_node->set_output_type(0, orig_type, m_node->get_output_partial_shape(0));
    }
};

template <typename BaseOp>
class TypeRelaxed : public BaseOp, public TypeRelaxedBase {
public:
    RTTI_DECLARATION;

    using BaseOp::BaseOp;

    TypeRelaxed(const BaseOp& base_op, element::Type use_type) : BaseOp(base_op), TypeRelaxedBase(use_type) {
        // reset all output ports as they contain references to original base_op instance
        // number of output ports will be restored in validate_and_infer_types
        BaseOp::set_output_size(0);
        BaseOp::update_inputs_after_copy_tmp();
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

template <typename BaseOp>
void TypeRelaxed<BaseOp>::validate_and_infer_types() {
    // std::cerr << "TypeRelaxed<BaseOp>::validate_and_infer_types from my template, name = " << BaseOp::get_name()
    //    << ", fiendly_name = " << BaseOp::get_friendly_name() << "\n";
    // std::cerr << "Description: " << BaseOp::description() << "\n";
    // Remember all input data types and reset them to m_output_data_type.
    std::vector<element::Type> input_types;
    for (size_t i = 0; i < BaseOp::get_input_size(); ++i) {
        input_types.push_back(BaseOp::get_input_element_type(i));
        BaseOp::get_input_tensor(i).set_tensor_type(m_output_data_type, BaseOp::get_input_partial_shape(i));
    }
    BaseOp::validate_and_infer_types();
    // Restore original input data types
    for (size_t i = 0; i < BaseOp::get_input_size(); ++i) {
        BaseOp::get_input_tensor(i).set_tensor_type(input_types[i], BaseOp::get_input_partial_shape(i));
    }
    // TODO Fix for multiple-output nodes as well
    BaseOp::set_output_type(0, m_output_data_type, BaseOp::get_output_partial_shape(0));
}


template <typename BaseOp>
RTTI_DEFINITION_1(BaseOp::get_type_info_static().name, TypeRelaxed<BaseOp>, BaseOp, 0)

template <typename BaseOp>
RTTI_DEFINITION_2(BaseOp::get_type_info_static().name, TypeRelaxed<BaseOp>, BaseOp, 0)

        template <typename BaseOp>
std::shared_ptr<Node> TypeRelaxed<BaseOp>::clone_with_new_inputs(const OutputVector& new_args) const {
    //auto clone = make_shared<TypeRelaxed<BaseOp>>();
    std::cerr << "TypeRelaxed<BaseOp>::copy_with_new_args from my template\n";
    // copy then modify inputs
    std::shared_ptr<Node> new_node = std::make_shared<TypeRelaxed<BaseOp>>((BaseOp&)(*this), m_output_data_type);
    for (size_t i = 0; i < new_node->get_input_size(); ++i) {
        new_node->input(i).replace_source_output(new_args[i]);
    }
    return new_node;
}


}  // namespace op
}  // namespace ngraph
