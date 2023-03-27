// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/core/any.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class DecoderMap : public ov::frontend::DecoderBase {
public:
    DecoderMap(std::shared_ptr<ov::frontend::DecoderBase> decoder,
               const std::map<std::string, ov::Any>& attrs,
               bool empty_name = false)
        : ov::frontend::DecoderBase(),
          m_attrs(attrs),
          m_decoder(std::move(decoder)),
          m_empty_name(empty_name) {}

    DecoderMap(std::shared_ptr<ov::frontend::DecoderBase> decoder,
               const std::map<std::string, ov::Any>& attrs,
               std::string type,
               bool empty_name = false)
        : ov::frontend::DecoderBase(),
          m_attrs(attrs),
          m_decoder(std::move(decoder)),
          m_type(type),
          m_empty_name(empty_name) {}

    /// \brief Get attribute value by name
    ///
    /// \param name Attribute name
    /// \return Shared pointer to appropriate value converted to openvino data type if it exists, 'nullptr' otherwise
    ov::Any get_attribute(const std::string& name) const override {
        if (m_attrs.count(name))
            return m_attrs.at(name);
        else
            return {};
    }

    /// \brief Get a number of inputs
    size_t get_input_size() const override {
        return m_decoder->get_input_size();
    }

    /// \brief Get a producer name and its output port index
    ///
    /// \param input_port_idx              Input port index by which data is consumed
    /// \param producer_name               A producer name
    /// \return producer_output_port_index Output port index from which data is generated
    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override {
        m_decoder->get_input_node(input_port_idx, producer_name, producer_output_port_index);
    }

    /// \brief Get a producer name and its output port index
    ///
    /// \param input_port_idx              Input port index by which data is consumed
    /// \param producer_name               A producer name
    /// \param producer_output_port_index  Output port index from which data is generated
    /// \param op_type_by_name             Map of operation name to their types
    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index,
                        const OpTypeByName& op_type_by_name) const override {
        FRONT_END_NOT_IMPLEMENTED("get_input_node method with op_type_by_name map is not implemented for TFL FE.");
    }

    /// \brief Get operation type
    const std::string& get_op_type() const override {
        if (m_type.empty())
            return m_decoder->get_op_type();
        return m_type;
    }

    /// \brief Get node name
    const std::string& get_op_name() const override {
        return m_empty_name ? empty_name : m_decoder->get_op_name();
    }

    /// \brief Destructor
    ~DecoderMap() = default;

private:
    std::map<std::string, ov::Any> m_attrs;
    std::shared_ptr<ov::frontend::DecoderBase> m_decoder;
    std::string m_type;
    const std::string empty_name;
    bool m_empty_name;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
