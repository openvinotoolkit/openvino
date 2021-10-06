// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/variant.hpp>

namespace ngraph {
namespace frontend {
namespace tf {

class DecoderBase {
public:
    /// \brief Get attribute value by name and requested type
    ///
    /// \param name Attribute name
    /// \param type_info Attribute type information
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual std::shared_ptr<Variant> get_attribute(const std::string& name, const VariantTypeInfo& type_info) const = 0;

    /// \brief Get a number of inputs
    virtual size_t get_input_size() const = 0;

    /// \brief Get a producer name and its output port index
    ///
    /// \param input_port_idx              Input port index by which data is consumed
    /// \param producer_name               A producer name
    /// \return producer_output_port_index Output port index from which data is generated
    virtual void get_input_node(const size_t input_port_idx,
                                std::string& producer_name,
                                size_t& producer_output_port_index) const = 0;

    /// \brief Get operation type
    virtual std::string get_op_type() const = 0;

    /// \brief Get node name
    virtual std::string get_op_name() const = 0;
};
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
