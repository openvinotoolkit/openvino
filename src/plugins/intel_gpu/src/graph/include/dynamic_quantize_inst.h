// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/dynamic_quantize.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

class DynamicQuantizeFuseParams : public NodeFuseParams {
public:
    DynamicQuantizeFuseParams(const std::vector<layout>& out_layouts, const dynamic_quantize::Attributes& attrs, size_t input_size)
        : NodeFuseParams(dynamic_quantize::type_id()),
          _out_layouts(out_layouts),
          _attrs(attrs),
          _input_size(input_size) {}

    std::vector<layout> _out_layouts;
    dynamic_quantize::Attributes _attrs;
    size_t _input_size;
};

template <>
struct typed_program_node<dynamic_quantize> : public typed_program_node_base<dynamic_quantize> {
    using parent = typed_program_node_base<dynamic_quantize>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<DynamicQuantizeFuseParams>(get_output_layouts(), get_primitive()->attrs, get_primitive()->input_size);
    }
};

using dynamic_quantize_node = typed_program_node<dynamic_quantize>;

template <>
class typed_primitive_inst<dynamic_quantize> : public typed_primitive_inst_base<dynamic_quantize> {
    using parent = typed_primitive_inst_base<dynamic_quantize>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(dynamic_quantize_node const& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(dynamic_quantize_node const& node, kernel_impl_params const& impl_params);

    // Internal function to be used from fakealignment
    template<typename ShapeType>
    static std::vector<layout> __calc_output_layouts(dynamic_quantize_node const& node,
                                                     const layout &act_layout,
                                                     const dynamic_quantize::Attributes& config);
    static std::string to_string(dynamic_quantize_node const& node);

    typed_primitive_inst(network& network, dynamic_quantize_node const& node);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using dynamic_quantize_inst = typed_primitive_inst<dynamic_quantize>;

}  // namespace cldnn
