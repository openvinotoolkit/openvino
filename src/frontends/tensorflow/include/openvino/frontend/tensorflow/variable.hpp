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

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
