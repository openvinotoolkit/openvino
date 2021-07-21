// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn/primitives/generic_primitive.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

    template <>
    struct typed_program_node<generic_primitive> : public typed_program_node_base<generic_primitive> {
        using parent = typed_program_node_base<generic_primitive>;

    public:
        using parent::parent;

        program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    };

    using generic_primitive_node = typed_program_node<generic_primitive>;

    template <>
    class typed_primitive_inst<generic_primitive> : public typed_primitive_inst_base<generic_primitive> {
        using parent = typed_primitive_inst_base<generic_primitive>;

    public:
        static layout calc_output_layout(generic_primitive_node const& node) {
            assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
                   "Output data type forcing is not supported for "
                   "generic_primitive_node!");
            layout output_layout = node.get_primitive()->output_layout;

            // if the output layout format was set to any,
            // it means the layer output format will be the same as the first input
            if (output_layout.format == format::any) {
                output_layout.format = node.get_dependency(0).get_output_layout().format;
            }
            return output_layout;
        }

        static std::string to_string(generic_primitive_node const& node);

    public:
        typed_primitive_inst(network_impl& network, generic_primitive_node const& node);
    };

    using generic_primitive_inst = typed_primitive_inst<generic_primitive>;

}  // namespace cldnn
