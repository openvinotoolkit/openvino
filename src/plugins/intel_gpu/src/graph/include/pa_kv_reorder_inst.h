// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "primitive_inst.h"

namespace cldnn {

using PaKVReorderInputIdx = cldnn::pa_kv_reorder::PaKVReorderInputIdx;

template <>
struct typed_program_node<pa_kv_reorder> : public typed_program_node_base<pa_kv_reorder> {
    using parent = typed_program_node_base<pa_kv_reorder>;

public:
    using parent::parent;
    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using pa_kv_reorder_node = typed_program_node<pa_kv_reorder>;

template <>
class typed_primitive_inst<pa_kv_reorder> : public typed_primitive_inst_base<pa_kv_reorder> {
    using parent = typed_primitive_inst_base<pa_kv_reorder>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(pa_kv_reorder_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(pa_kv_reorder_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(pa_kv_reorder_node const& node);

    typed_primitive_inst(network& network, pa_kv_reorder_node const& node);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using pa_kv_reorder_inst = typed_primitive_inst<pa_kv_reorder>;

}  // namespace cldnn
