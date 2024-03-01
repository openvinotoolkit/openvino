// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// this class describes TensorFlow table produced by operations tf.raw_ops.HashTable, tf.raw_ops.HashTableV2,
// tf.raw_ops.MutableHashTable and stores a dictionary of keys mapped to values
// Objects of this class is fed to Lookup* operations for initialization and searching values by keys
// Types of keys and values can be different
class HashTable : public Variable {
public:
    using Ptr = std::shared_ptr<HashTable>;
    OPENVINO_OP("TFHashTable", "ov::frontend::tensorflow", Variable);

    HashTable(const std::string& name,
              const ov::element::Type& key_type,
              const ov::element::Type& value_type,
              const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : Variable(name, decoder),
          m_key_type(key_type),
          m_value_type(value_type) {
        validate_and_infer_types();
    }

    HashTable(const HashTable& other, const ov::Output<ov::Node>& keys, const ov::Output<ov::Node>& values)
        : HashTable(other) {
        m_keys = keys;
        m_values = values;
        // reset names of tensor corresponding to variable value
        // that is because variable can have multiple values during inference
        m_keys.set_names({});
        m_values.set_names({});
        m_is_initialized = true;
        ++m_init_counter;
    }

    // it must be used only for cloning
    // other ways are illegal
    HashTable(const std::string& name,
              const ov::element::Type& key_type,
              const ov::element::Type& value_type,
              const ov::Output<ov::Node>& keys,
              const ov::Output<ov::Node>& values,
              bool is_initialized,
              uint64_t init_counter,
              const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : Variable(name, decoder),
          m_key_type(key_type),
          m_value_type(value_type),
          m_keys(keys),
          m_values(values) {
        m_init_counter = init_counter;
        m_is_initialized = is_initialized;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // this is a type of resource so its shape and type is not applicable
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto hash_table_node = std::make_shared<HashTable>(m_name,
                                                           m_key_type,
                                                           m_value_type,
                                                           m_keys,
                                                           m_values,
                                                           m_is_initialized,
                                                           m_init_counter,
                                                           m_decoder);
        hash_table_node->set_attrs(get_attrs());
        return hash_table_node;
    }

    ov::Output<ov::Node> get_values() const {
        FRONT_END_GENERAL_CHECK(m_is_initialized,
                                "[TensorFlow Frontend] internal error: get_values() is called for uninitialized table");
        return m_values;
    }

    ov::Output<ov::Node> get_keys() const {
        FRONT_END_GENERAL_CHECK(m_is_initialized,
                                "[TensorFlow Frontend] internal error: get_values() is called for uninitialized table");
        return m_keys;
    }

    ov::Output<ov::Node> get_value() override {
        return output(0);
    }

    ov::element::Type get_key_type() const {
        return m_key_type;
    }

private:
    ov::element::Type m_key_type;
    ov::element::Type m_value_type;
    ov::Output<ov::Node> m_keys;
    ov::Output<ov::Node> m_values;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
