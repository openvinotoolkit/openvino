// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/fake_convert.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<fake_convert> : public typed_program_node_base<fake_convert> {
    using parent = typed_program_node_base<fake_convert>;
    typed_program_node(const std::shared_ptr<fake_convert> prim, program& prog)
        : parent(prim, prog), destination_type(prim->destination_type) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& scale() const { return get_dependency(1); }
    program_node& shift() const { return get_dependency(2); }
    bool has_shift() const { return (get_dependencies().size() == 3); }

    ov::element::Type get_destination_type() const { return destination_type; }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

private:
    ov::element::Type destination_type;
};

using fake_convert_node = typed_program_node<fake_convert>;

template <>
class typed_primitive_inst<fake_convert> : public typed_primitive_inst_base<fake_convert> {
    using parent = typed_primitive_inst_base<fake_convert>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(fake_convert_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(fake_convert_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(fake_convert_node const& node);

    typed_primitive_inst(network& network, fake_convert_node const& node);
};

using fake_convert_inst = typed_primitive_inst<fake_convert>;
}  // namespace cldnn
