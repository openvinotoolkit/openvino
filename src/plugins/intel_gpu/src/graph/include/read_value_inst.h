// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "assign_inst.h"
#include "intel_gpu/primitives/read_value.hpp"
#include "primitive_inst.h"

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
        const auto default_layout = desc->output_layout;
        auto out_layout = impl_param.state_layout.value_or(default_layout);
        if (out_layout.is_dynamic() && desc->input_size() > 0) {
            out_layout = impl_param.get_input_layout(0);
        }
        return { out_layout };
    }

    static layout calc_output_layout(const read_value_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const read_value_node& node);

    typed_primitive_inst(network& network, const read_value_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}

    void save(cldnn::BinaryOutputBuffer& ob) const override;
    void load(cldnn::BinaryInputBuffer& ib) override;

    void update_output_memory() override;

protected:
    void on_execute() override;
};

using read_value_inst = typed_primitive_inst<read_value>;

} // namespace cldnn
