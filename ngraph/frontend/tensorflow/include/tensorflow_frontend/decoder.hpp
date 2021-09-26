// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/variant.hpp>

namespace ngraph {
namespace frontend {

class DecoderBase {
public:
    /// \brief Get attribute value by name and requested type
    ///
    /// \param name Attribute name
    /// \param type_info Attribute type information
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual std::shared_ptr<Variant> get_attribute(const std::string& name, const VariantTypeInfo& type_info) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual void get_input_node(const size_t input_port_idx,
                                std::string& producer_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::string get_op_type() const = 0;

    virtual std::string get_op_name() const = 0;
};

}  // namespace frontend
}  // namespace ngraph
