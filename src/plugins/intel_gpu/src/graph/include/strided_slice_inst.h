// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/strided_slice.hpp"
#include "primitive_inst.h"

#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<strided_slice> : public typed_program_node_base<strided_slice> {
    using parent = typed_program_node_base<strided_slice>;
    typed_program_node(const std::shared_ptr<strided_slice> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2, 3}; }
};

using strided_slice_node = typed_program_node<strided_slice>;

template <>
class typed_primitive_inst<strided_slice> : public typed_primitive_inst_base<strided_slice> {
    using parent = typed_primitive_inst_base<strided_slice>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(strided_slice_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(strided_slice_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(strided_slice_node const& node);

    typed_primitive_inst(network& network, strided_slice_node const& desc);

    void update_output_memory() override;

private:
    void on_execute() override;
};

using strided_slice_inst = typed_primitive_inst<strided_slice>;
}  // namespace cldnn
