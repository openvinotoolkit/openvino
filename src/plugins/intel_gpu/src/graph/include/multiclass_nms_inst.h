// Copyright (C) 2022-2024 Intel Corporation
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
        return get_primitive()->input_size() == 3;
    }

    const program_node& roisnum() const {
        OPENVINO_ASSERT(has_roisnum(), "[GPU] rois_num not found");
        return get_dependency(2);
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
    static std::string to_string(const multiclass_nms_node& node);

    typed_primitive_inst(network& network, const multiclass_nms_node& node) : parent(network, node) {}
};

using multiclass_nms_inst = typed_primitive_inst<multiclass_nms>;

}  // namespace cldnn
