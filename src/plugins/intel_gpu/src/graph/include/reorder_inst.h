// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/reorder.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/reorder/reorder_kernel_base.h"
#include "kernel_selector/common/tensor_type.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<reorder> : public typed_program_node_base<reorder> {
    using parent = typed_program_node_base<reorder>;

public:
    typed_program_node(const std::shared_ptr<reorder> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

    size_t inputs_count() const { return get_primitive()->input.size(); }
    program_node& mean_nv12() const { return get_dependency(2); }
    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    program_node& mean() const { return get_dependency(1); }

    bool has_mean() const { return !typed_desc()->mean.empty(); }

    bool requires_reinterpret() const { return req_reinterpr; }
    void requires_reinterpret(bool val) { req_reinterpr = (optimized && val); }

    void set_input_layout(layout const& lo) { input_layout = lo; }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        kernel_selector::DataLayout ks_input_layout = convert_data_tensor(input_layout).GetLayout();
        kernel_selector::DataLayout ks_output_layout = convert_data_tensor(get_output_layout()).GetLayout();
        return std::make_shared<kernel_selector::reorder_fuse_params>(ks_input_layout, ks_output_layout);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

private:
    bool req_reinterpr = false;
    layout input_layout = layout(data_types::f32, format::bfyx, { 0, 0, 0, 0 });
};

using reorder_node = typed_program_node<reorder>;

template <>
class typed_primitive_inst<reorder> : public typed_primitive_inst_base<reorder> {
    using parent = typed_primitive_inst_base<reorder>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reorder_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(reorder_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reorder_node const& node);

public:
    typed_primitive_inst(network& network, reorder_node const& node);
    memory::ptr mean_nv12_memory() const { return dep_memory_ptr(2); }
    memory::ptr mean_memory() const { return dep_memory_ptr(1); }

    bool has_mean() const { return !argument.mean.empty(); }

private:
    void on_execute() override;
    void reuse_input();
};

using reorder_inst = typed_primitive_inst<reorder>;

}  // namespace cldnn
