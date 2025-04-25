// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/crop.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<crop> : public typed_program_node_base<crop> {
private:
    using parent = typed_program_node_base<crop>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<crop> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        std::vector<size_t> vec;
        for (size_t i  = 1; i < get_dependencies().size(); i++) {
            vec.push_back(i);
        }
        return vec;
    }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->input_offsets.reserve(1);
        params->input_offsets.push_back(get_primitive()->offsets);
        return params;
    }
};

using crop_node = typed_program_node<crop>;

template <>
class typed_primitive_inst<crop> : public typed_primitive_inst_base<crop> {
    using parent = typed_primitive_inst_base<crop>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const crop_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(crop_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(crop_node const& node);
    typed_primitive_inst(network& network, crop_node const& node);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using crop_inst = typed_primitive_inst<crop>;
}  // namespace cldnn
