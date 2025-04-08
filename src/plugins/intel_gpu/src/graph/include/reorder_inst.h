// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/reorder.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

class ReorderFuseParams : public NodeFuseParams {
public:
    ReorderFuseParams(const layout& in, const layout& out) : NodeFuseParams(reorder::type_id()), _in(in), _out(out) {}

    layout _in;
    layout _out;
};

template <>
struct typed_program_node<reorder> : public typed_program_node_base<reorder> {
    using parent = typed_program_node_base<reorder>;

public:
    typed_program_node(const std::shared_ptr<reorder> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

    program_node& mean_nv12() const { return get_dependency(2); }
    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    program_node& mean() const { return get_dependency(1); }

    bool has_mean() const { return !typed_desc()->mean.empty(); }

    bool requires_reinterpret() const { return req_reinterpr; }
    void requires_reinterpret(bool val) { req_reinterpr = (optimized && val); }

    void set_input_layout(layout const& lo) { input_layout = lo; }

    bool is_type_conversion_only() const {
        auto in_layout = get_input_layout();
        auto out_layout = get_output_layout();
        bool only_precision_changed = in_layout.data_type != out_layout.data_type &&
                                      in_layout.format == out_layout.format &&
                                      in_layout.data_padding == out_layout.data_padding;
        if (is_dynamic()) {
            only_precision_changed &= in_layout.get_partial_shape().rank() == out_layout.get_partial_shape().rank();
        } else {
            only_precision_changed &= in_layout.get_partial_shape() == out_layout.get_partial_shape();
        }

        return only_precision_changed && is_simple_reorder() && typed_desc()->truncate;
    }

    bool is_simple_reorder() const {
        return !has_fused_primitives() &&
               !has_mean() &&
               get_primitive()->subtract_per_feature.empty() &&
               !get_primitive()->weights_reorder_params;
    }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<ReorderFuseParams>(input_layout, get_output_layout());
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
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reorder_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(reorder_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reorder_node const& node);

public:
    typed_primitive_inst(network& network);
    typed_primitive_inst(network& network, reorder_node const& node);

    memory::ptr mean_nv12_memory() const { return dep_memory_ptr(2); }
    memory::ptr mean_memory() const { return dep_memory_ptr(1); }

    bool has_mean() const { return !get_typed_desc<reorder>()->mean.empty(); }

    void update_output_memory() override;
    bool requires_reinterpret() const {
        auto req_reinterpr = _req_reinterpr;
        if (input_memory().get_layout() != _impl_params->get_output_layout()) {
            req_reinterpr = true;
        }
        return req_reinterpr;
    }

private:
    void on_execute() override;

    bool _req_reinterpr = false;
};

using reorder_inst = typed_primitive_inst<reorder>;

}  // namespace cldnn
