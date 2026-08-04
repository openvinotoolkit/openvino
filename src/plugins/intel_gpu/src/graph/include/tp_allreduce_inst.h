// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_TP_GPU

#include "intel_gpu/primitives/tp_allreduce.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<tp_allreduce> : public typed_program_node_base<tp_allreduce> {
    using parent = typed_program_node_base<tp_allreduce>;
    typed_program_node(const std::shared_ptr<tp_allreduce> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using tp_allreduce_node = typed_program_node<tp_allreduce>;

template <>
class typed_primitive_inst<tp_allreduce> : public typed_primitive_inst_base<tp_allreduce> {
    using parent = typed_primitive_inst_base<tp_allreduce>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(tp_allreduce_node const& node,
                                                   const kernel_impl_params& impl_param);
    static layout calc_output_layout(tp_allreduce_node const& node,
                                     kernel_impl_params const& impl_param);
    static std::string to_string(tp_allreduce_node const& node);

    typed_primitive_inst(network& network, tp_allreduce_node const& desc);
};

using tp_allreduce_inst = typed_primitive_inst<tp_allreduce>;

}  // namespace cldnn

#endif  // ENABLE_TP_GPU
