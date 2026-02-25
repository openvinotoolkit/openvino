// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<multiclass_nms> : public typed_program_node_base<multiclass_nms> {
    using parent = typed_program_node_base<multiclass_nms>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {}

    const program_node& input() const {
        return boxes();
    }

    const program_node& boxes() const {
        return get_dependency(0);
    }

    const program_node& scores() const {
        return get_dependency(1);
    }

    bool has_roisnum() const {
        return get_primitive()->has_roisnum;
    }

    const program_node& roisnum() const {
        if (!get_primitive()->has_roisnum)
            throw std::runtime_error("there is no roisnum input");
        return get_dependency(2);
    }

    const program_node& output_selected_indices() const {
        return get_dependency(input_count());
    }
    const program_node& output_selected_num() const {
        return get_dependency(input_count() + 1);
    }

private:
    int input_count() const {
        return 2 + (get_primitive()->has_roisnum ? 1 : 0);
    }
};

using multiclass_nms_node = typed_program_node<multiclass_nms>;

template <>
class typed_primitive_inst<multiclass_nms> : public typed_primitive_inst_base<multiclass_nms> {
    using parent = typed_primitive_inst_base<multiclass_nms>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(multiclass_nms_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const multiclass_nms_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const multiclass_nms_node& node);

    typed_primitive_inst(network& network, const multiclass_nms_node& node) : parent(network, node) {}

    memory::ptr output_indices_memory() const {
        return dep_memory_ptr(dependencies().size() - 2);
    }
    memory::ptr output_num_memory() const {
        return dep_memory_ptr(dependencies().size() - 1);
    }
};

using multiclass_nms_inst = typed_primitive_inst<multiclass_nms>;

}  // namespace cldnn
