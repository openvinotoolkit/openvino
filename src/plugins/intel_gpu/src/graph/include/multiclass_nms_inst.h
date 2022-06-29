// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<multiclass_nms> : public typed_program_node_base<multiclass_nms> {
    using parent = typed_program_node_base<multiclass_nms>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          has_roisnum_(this->get_primitive()->dependencies().size() == 7) {}

    program_node& input() const {
        return boxes();
    }

    program_node& boxes() const {
        return get_dependency(0);
    }

    program_node& scores() const {
        return get_dependency(1);
    }

    bool has_roisnum() const {
        return has_roisnum_;
    }

    program_node& roisnum() const {
        if (!has_roisnum_)
            throw std::runtime_error("there is no roisnum input");
        return get_dependency(2);
    }

    program_node& output_indices() const {
        return get_dependency(input_count());
    }
    program_node& output_num() const {  // FIXME opoluektov: don't like the name
        return get_dependency(input_count() + 1);
    }

private:
    int input_count() const {
        return 2 + static_cast<int>(has_roisnum_);
    }

    bool has_roisnum_;
};

using multiclass_nms_node = typed_program_node<multiclass_nms>;

template <>
class typed_primitive_inst<multiclass_nms> : public typed_primitive_inst_base<multiclass_nms> {
    using parent = typed_primitive_inst_base<multiclass_nms>;

public:
//    template<typename ShapeType>
//    static std::vector<layout> calc_output_layouts(const multiclass_nms_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(multiclass_nms_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(const multiclass_nms_node& node);

    typed_primitive_inst(network& network, const multiclass_nms_node& node) : parent(network, node) {}

    memory::ptr output_indices_memory() const {
        return dep_memory_ptr(node.get_dependencies().size() - 1);
    }
    memory::ptr output_num_memory() const {
        // FIXME opoluektov: harmonize impl with typed_program_node<multiclass_nms>
        return dep_memory_ptr(node.get_dependencies().size() - 2);
    }
};

using multiclass_nms_inst = typed_primitive_inst<multiclass_nms>;

}  // namespace cldnn
