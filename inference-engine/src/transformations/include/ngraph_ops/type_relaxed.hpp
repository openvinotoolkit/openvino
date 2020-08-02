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

/// A base class for templated TypeRelaxed that maintains overriden input types and outtypes for an operations.
class TRANSFORMATIONS_API TypeRelaxedBase {
public:
    virtual ~TypeRelaxedBase();

    TypeRelaxedBase(
            const std::vector<element::Type>& _input_data_types = {},
            const std::vector<element::Type>& _output_data_types = {}) :
            m_input_data_types(_input_data_types),
            m_output_data_types(_output_data_types) {
    }

    /// \return Data type that will be set for output with a given index outputIndex.
    /// If output with a specified index outputIndex hasn't been set before, element::undefined will returned.
    /// Undefined means no type override happens for a given outputIndex and it will deduced as original
    /// operation defineds in its infer function.
    const element::Type& get_overriden_output_type(size_t outputIndex = 0) const {
        if (outputIndex >= m_output_data_types.size()) {
            return element::undefined;
        }
        return m_output_data_types[outputIndex];
    }

    /// Set data type that overrides the original data type for output port with outputIndex index
    /// In case if outputIndex is out of range of known outputs (and this class cannot detect
    /// the real number of outputs for original operation), the number of overriden outputs
    /// is changed according to a given outputIndex value.
    void set_overriden_output_type(const element::Type& element_type, size_t outputIndex = 0) {
        if (outputIndex >= m_output_data_types.size()) {
            m_output_data_types.resize(outputIndex + 1, element::undefined);
        }
        m_output_data_types[outputIndex] = element_type;
    }

    /// \return Data type that will be set for input when original shape/type inference function is called.
    /// If index inputIndex hasn't been set before, element::undefined will returned. Undefined means that
    /// the type from input tensor descriptor is used for a given index.
    const element::Type& get_origin_input_type(size_t inputIndex = 0) const {
        if (inputIndex >= m_input_data_types.size()) {
            return element::undefined;
        }
        return m_input_data_types[inputIndex];
    }

    /// Set data type that overrides the original data type for input port with inputIndex index.
    /// In case if inputIndex is out of range of known inputs (and this class cannot detect
    /// the real number of inputs for original operation), the number of overriden inputs
    /// is changed according to a given inputIndex value. All new entries except one added
    /// at inputIndex positoin are undefined.
    void set_origin_input_type(const element::Type& element_type, size_t inputIndex = 0) {
        if (inputIndex >= m_input_data_types.size()) {
            m_input_data_types.resize(inputIndex + 1, element::undefined);
        }
        m_input_data_types[inputIndex] = element_type;
    }

protected:
    // Data types that are used for parent shape/type infer function input ports
    // to infer output data types
    std::vector<element::Type> m_input_data_types;
    std::vector<element::Type> m_output_data_types;
};

class TemporaryReplaceOutputType {
    std::shared_ptr<Node> m_node;
    element::Type orig_type;
public:
    TemporaryReplaceOutputType(std::shared_ptr<Node> node, element::Type tmp_type) : m_node(node) {
        orig_type = node->get_output_element_type(0);
        m_node->set_output_type(0, tmp_type, m_node->get_output_partial_shape(0));
    }

    operator std::shared_ptr<Node>() const {
        return m_node;
    }

    std::shared_ptr<Node> get() const {
        return m_node;
    }

    ~TemporaryReplaceOutputType() {
        m_node->set_output_type(0, orig_type, m_node->get_output_partial_shape(0));
    }
};

template <typename BaseOp>
class TypeRelaxed : public BaseOp, public TypeRelaxedBase {
public:
    NGRAPH_RTTI_DECLARATION;

    using BaseOp::BaseOp;

    TypeRelaxed() = default;

    TypeRelaxed(
            const BaseOp& base_op,
            element::Type overriden_type) :
            TypeRelaxed(base_op,
                    std::vector<element::Type>(base_op.get_input_size(), overriden_type),
                    std::vector<element::Type>(base_op.get_output_size(), overriden_type)) {
    }

    explicit TypeRelaxed(
            const BaseOp& base_op,
            const std::vector<element::Type>& _input_data_types = {},
            const std::vector<element::Type>& _output_data_types = {}) :
            BaseOp(base_op), TypeRelaxedBase(_input_data_types, _output_data_types) {
        init();
    }

    /// Creating a new TypeRelaxed operation by calling one of the original op ctors forwarding arguments directly.
    template <typename ... Args>
    TypeRelaxed(
            const std::vector<element::Type>& _input_data_types,
            const std::vector<element::Type>& _output_data_types,
            Args&& ... args) :
            BaseOp(std::forward<Args>(args)...), TypeRelaxedBase(_input_data_types, _output_data_types) {
        init();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    void init() {
        // reset all output ports as they contain references to original base_op instance
        // number of output ports will be restored in validate_and_infer_types
        BaseOp::set_output_size(0);
        BaseOp::update_inputs_after_copy_tmp(); // ugly?
        validate_and_infer_types();
    }
};

template <typename BaseOp>
void TypeRelaxed<BaseOp>::validate_and_infer_types() {
    // Remember all input data types and reset them to m_output_data_type.
    std::vector<element::Type> old_input_types;
    for (size_t i = 0; i < BaseOp::get_input_size(); ++i) {
        old_input_types.push_back(BaseOp::get_input_element_type(i));
        auto origin_input_type = get_origin_input_type(i);
        if (origin_input_type != element::undefined) {
            BaseOp::get_input_tensor(i).set_tensor_type(origin_input_type, BaseOp::get_input_partial_shape(i));
        }
    }

    BaseOp::validate_and_infer_types();

    // Restore original input data types
    for (size_t i = 0; i < BaseOp::get_input_size(); ++i) {
        BaseOp::get_input_tensor(i).set_tensor_type(old_input_types[i], BaseOp::get_input_partial_shape(i));
    }

    // Override (some) output types
    for (size_t i = 0; i < BaseOp::get_output_size(); ++i) {
        auto overriden_output_type = get_overriden_output_type(i);
        if (overriden_output_type != element::undefined) {
            BaseOp::set_output_type(0, overriden_output_type, BaseOp::get_output_partial_shape(i));
        }
    }
}



template <typename BaseOp>
std::shared_ptr<Node> TypeRelaxed<BaseOp>::clone_with_new_inputs(const OutputVector& new_args) const {
    //auto clone = make_shared<TypeRelaxed<BaseOp>>();
    std::cerr << "TypeRelaxed<BaseOp>::copy_with_new_args from my template\n";
    // copy then modify inputs
    std::shared_ptr<Node> new_node = std::make_shared<TypeRelaxed<BaseOp>>((BaseOp&)(*this), m_input_data_types, m_output_data_types);
    for (size_t i = 0; i < new_node->get_input_size(); ++i) {
        new_node->input(i).replace_source_output(new_args[i]);
    }
    return new_node;
}

template <typename BaseOp>
const ::ngraph::Node::type_info_t& TypeRelaxed<BaseOp>::get_type_info() const { return type_info; }
template <typename BaseOp>
    const ::ngraph::Node::type_info_t& TypeRelaxed<BaseOp>::get_type_info_static() { return type_info; }
    template <typename BaseOp>
    const ::ngraph::Node::type_info_t TypeRelaxed<BaseOp>::type_info{
        // TODO: Incorrect name 'TypeRelaxed', should be different for various BaseOp
        "TypeRelaxed", 0, &BaseOp::type_info};

}  // namespace op
}  // namespace ngraph
