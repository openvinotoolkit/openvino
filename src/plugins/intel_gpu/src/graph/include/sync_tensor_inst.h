// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/sync_tensor.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<sync_tensor> : public typed_program_node_base<sync_tensor> {
private:
    using parent = typed_program_node_base<sync_tensor>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using sync_tensor_node = typed_program_node<sync_tensor>;

template<>
class typed_primitive_inst<sync_tensor> : public typed_primitive_inst_base<sync_tensor> {
    using parent = typed_primitive_inst_base<sync_tensor>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(sync_tensor_node const& /*node*/, const kernel_impl_params& impl_param);

    static layout calc_output_layout(const sync_tensor_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const sync_tensor_node& node);

    typed_primitive_inst(network& network, const sync_tensor_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}
    void update_output_memory() override;

protected:
    void on_execute() override;
};

using sync_tensor_inst = typed_primitive_inst<sync_tensor>;

} // namespace cldnn