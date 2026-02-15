// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lora.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<lora> : public typed_program_node_base<lora> {
    using parent = typed_program_node_base<lora>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using lora_node = typed_program_node<lora>;

template <>
class typed_primitive_inst<lora> : public typed_primitive_inst_base<lora> {
    using parent = typed_primitive_inst_base<lora>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const lora_node& /*node*/, const kernel_impl_params& impl_params) {
        return forward_input0_shape<ShapeType>(impl_params);
    }
    static layout calc_output_layout(const lora_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const lora_node& node);

    typed_primitive_inst(network& network, const lora_node& node);

    void update_output_memory() override;

private:
    void on_execute() override;
};

using lora_inst = typed_primitive_inst<lora>;

}  // namespace cldnn
