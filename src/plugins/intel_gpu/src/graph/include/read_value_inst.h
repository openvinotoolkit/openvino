// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/read_value.hpp"
#include "primitive_inst.h"
#include "variable.hpp"

namespace cldnn {

template <>
struct typed_program_node<read_value> : public typed_program_node_base<read_value> {
private:
    using parent = typed_program_node_base<read_value>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using read_value_node = typed_program_node<read_value>;

template<>
class typed_primitive_inst<read_value> : public typed_primitive_inst_base<read_value>, public memory_state::variable {
    using parent = typed_primitive_inst_base<read_value>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(read_value_node const& /*node*/, const kernel_impl_params& impl_param) {
        auto desc = impl_param.typed_desc<read_value>();
        std::vector<layout> output_layouts;

        for (size_t i = 0; i < desc->num_outputs; i++) {
            const auto& default_layout = desc->output_layouts[i];
            output_layouts.push_back(impl_param.state_layouts.size() > i ? impl_param.state_layouts[i] : default_layout);
        }

        return output_layouts;
    }

    static layout calc_output_layout(const read_value_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const read_value_node& node);

    typed_primitive_inst(network& network, const read_value_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}

    void update_output_memory() override;

protected:
    void on_execute() override;
};

using read_value_inst = typed_primitive_inst<read_value>;

} // namespace cldnn
