// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class Variable : public ov::op::util::FrameworkNode {
public:
    using Ptr = std::shared_ptr<Variable>;
    OPENVINO_OP("TFVariable", "ov::frontend::tensorflow", ::ov::op::util::FrameworkNode);

    Variable(const std::string& name, const std::shared_ptr<DecoderBase>& decoder)
        : ov::op::util::FrameworkNode(ov::OutputVector{}, 1),
          m_name(name),
          m_shape(ov::Shape{}),
          m_type(ov::element::dynamic),
          m_decoder(decoder),
          m_is_initialized(false),
          m_init_counter(0) {
        validate_and_infer_types();
    }

    Variable(const std::string& name,
             const ov::Shape& shape,
             const ov::element::Type& type,
             const std::shared_ptr<DecoderBase>& decoder)
        : ov::op::util::FrameworkNode(ov::OutputVector{}, 1),
          m_name(name),
          m_shape(shape),
          m_type(type),
          m_decoder(decoder),
          m_is_initialized(false),
          m_init_counter(0) {
        validate_and_infer_types();
    }

    Variable(const std::string& name,
             const ov::Shape& shape,
             const ov::element::Type& type,
             const ov::Output<ov::Node>& value,
             const std::shared_ptr<DecoderBase>& decoder)
        : Variable(name, shape, type, decoder) {
        m_value = value;
        // reset names of tensor corresponding to variable value
        // that is because variable can have multiple values during inference
        m_value.set_names({});
        m_is_initialized = true;
        ++m_init_counter;
    }

    Variable(const Variable& other, const ov::Output<ov::Node>& value) : Variable(other) {
        m_value = value;
        // reset names of tensor corresponding to variable value
        // that is because variable can have multiple values during inference
        m_value.set_names({});
        m_is_initialized = true;
        ++m_init_counter;
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_type, m_shape);
    }

    bool is_initialized() const {
        return m_is_initialized;
    }

    virtual ov::Output<ov::Node> get_value() {
        FRONT_END_GENERAL_CHECK(
            m_is_initialized,
            "[TensorFlow Frontend] internal error: get_value() is called for uninitialized variable");
        return m_value;
    }

    std::string get_name() const {
        return m_name;
    }

    uint64_t get_init_counter() const {
        return m_init_counter;
    }

    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto new_variable = std::make_shared<Variable>(*this);
        new_variable->set_attrs(get_attrs());
        return new_variable;
    }

protected:
    std::string m_name;
    ov::Shape m_shape;
    ov::element::Type m_type;
    std::shared_ptr<DecoderBase> m_decoder;
    bool m_is_initialized;
    ov::Output<ov::Node> m_value;
    // this member is used to select the latest state of Variable
    uint64_t m_init_counter;
};

// a container of Variables state for each operation node in a graph
class VariableMap {
public:
    using Ptr = std::shared_ptr<VariableMap>;
    bool get_variable_state(const std::string& node_name,
                            const std::string& variable_name,
                            Variable::Ptr& found_variable) const {
        if (m_variables_state.count(node_name) > 0) {
            for (const auto& variable : m_variables_state.at(node_name)) {
                if (variable && variable->get_name() == variable_name && variable->is_initialized()) {
                    found_variable = variable;
                    return true;
                }
            }
        } else {
            return false;
        }
        return false;
    }

    void initialize_variable_state_map_for_node(const std::vector<std::string>& control_dependencies,
                                                const std::vector<std::string>& data_dependencies,
                                                const std::string& node_name) {
        m_variables_state[node_name] = {};
        for (const auto& dependency : control_dependencies) {
            for (const auto& dependency_variable : m_variables_state[dependency]) {
                update_variable_state_map_for_node(node_name, dependency_variable);
            }
        }

        for (const auto& dependency : data_dependencies) {
            for (const auto& dependency_variable : m_variables_state[dependency]) {
                update_variable_state_map_for_node(node_name, dependency_variable);
            }
        }
    }

    void update_variable_state_map_for_node(const std::string& node_name, const Variable::Ptr& update_variable) {
        FRONT_END_GENERAL_CHECK(
            update_variable && update_variable->is_initialized(),
            "[TensorFlow Frontend] internal error: variable maps must be updated with initialized variable");
        auto variable_name = update_variable->get_name();

        size_t remove_ind = 0;
        bool remove_old_variable = false;
        bool found_better = false;
        // remove old variable state if exists
        for (size_t ind = 0; ind < m_variables_state[node_name].size(); ++ind) {
            auto checked_variable = m_variables_state[node_name][ind];
            if (checked_variable->get_name() == variable_name && checked_variable->is_initialized() &&
                checked_variable->get_init_counter() < update_variable->get_init_counter()) {
                remove_ind = ind;
                remove_old_variable = true;
                break;
            } else if (checked_variable->get_name() == variable_name && checked_variable->is_initialized() &&
                       checked_variable->get_init_counter() >= update_variable->get_init_counter()) {
                found_better = true;
            }
        }

        if (remove_old_variable) {
            // update the variable map with new variable
            m_variables_state[node_name].erase(m_variables_state[node_name].begin() + remove_ind);
        }

        if (!found_better) {
            m_variables_state[node_name].push_back(update_variable);
        }
    }

private:
    std::unordered_map<std::string, std::vector<Variable::Ptr>> m_variables_state;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
