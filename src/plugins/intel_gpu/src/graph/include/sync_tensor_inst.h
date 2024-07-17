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
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {
            if (prog.get_config().get_context_for_tp().size() > 1) { // only apply for dynamic models for now
                w_rank = prog.get_config().subStreamExecConfig.get_rank()[0];
                w_size = prog.get_config().get_context_for_tp().size();
                GPU_DEBUG_TRACE_DETAIL << "Apply TP rank " << w_rank << " : " << w_size << std::endl;
            }
        }
    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        // runtime decision if needs to apply TP based on different FC inputs/weights
        if (is_dynamic()) {
            params->w_rank = w_rank;
            params->w_size = w_size;
        }
        return params;
    }
    int w_rank = -1;
    size_t w_size = 1; // default 1 process, world size as 1
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
    std::vector<uint64_t> host_sync_times;
    std::vector<uint64_t> host_copy_times;

protected:
    void on_execute() override;
};

using sync_tensor_inst = typed_primitive_inst<sync_tensor>;

} // namespace cldnn