// Copyright (C) 2018-2023 Intel Corporation
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
    ReorderFuseParams(layout in, layout out) : NodeFuseParams(reorder::type_id()), _in(in), _out(out) {}

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

    size_t inputs_count() const { return get_primitive()->input.size(); }
    program_node& mean_nv12() const { return get_dependency(2); }
    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    program_node& mean() const { return get_dependency(1); }

    bool has_mean() const { return !typed_desc()->mean.empty(); }

    bool requires_reinterpret() const { return req_reinterpr; }
    void requires_reinterpret(bool val) { req_reinterpr = (optimized && val); }

    void set_input_layout(layout const& lo) { input_layout = lo; }

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
    typed_primitive_inst(network& network, reorder_node const& node);
    memory::ptr mean_nv12_memory() const { return dep_memory_ptr(2); }
    memory::ptr mean_memory() const { return dep_memory_ptr(1); }

    bool has_mean() const { return !get_typed_desc<reorder>()->mean.empty(); }

    void update_output_memory() override;
    bool requires_reinterpret() const { return _req_reinterpr; }

    void save(cldnn::BinaryOutputBuffer& ob) const override;
    void load(cldnn::BinaryInputBuffer& ib) override;

private:
    void on_execute() override;
    void reuse_input();

    bool _req_reinterpr = false;
};

using reorder_inst = typed_primitive_inst<reorder>;

}  // namespace cldnn
