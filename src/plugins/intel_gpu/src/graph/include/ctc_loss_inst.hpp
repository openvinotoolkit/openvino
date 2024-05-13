// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/ctc_loss.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<ctc_loss> : typed_program_node_base<ctc_loss> {
    using parent = typed_program_node_base<ctc_loss>;
    using parent::parent;

    program_node& input(size_t index) const {
        return get_dependency(index);
    }
};

using ctc_loss_node = typed_program_node<ctc_loss>;

template <>
class typed_primitive_inst<ctc_loss> : public typed_primitive_inst_base<ctc_loss> {
public:
    using parent = typed_primitive_inst_base<ctc_loss>;
    using parent::parent;
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(ctc_loss_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const ctc_loss_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const ctc_loss_node& node);
};

using ctc_loss_inst = typed_primitive_inst<ctc_loss>;

}  // namespace cldnn
