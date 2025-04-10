// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(condition)

const size_t idx_branch_true    = 0;
const size_t idx_branch_false   = 1;

static std::map<primitive_id, layout> get_out_layout_map(cldnn::program::ptr prog) {
    std::map<primitive_id, layout> out_layout_map;
    for (auto& o : prog->get_outputs()) {
        out_layout_map.insert({o->id(), o->get_output_layout()});
    }
    return out_layout_map;
}

static std::map<primitive_id, layout> get_out_layout_map(cldnn::network::ptr net) {
    std::map<primitive_id, layout> out_layout_map;
    for (auto& o : net->get_outputs()) {
        out_layout_map.insert({o->id(), o->get_output_layout()});
    }
    return out_layout_map;
}

static std::vector<layout> get_output_layouts(std::map<primitive_id, layout>&& outputs, const std::map<size_t, cldnn::primitive_id> &io_output_map) {
    std::vector<layout> out_layouts;
    for (auto out : outputs) {
        for (auto& io_output : io_output_map) {
            auto inner_prim_id = io_output.second;
            if (out.first == inner_prim_id) {
                out_layouts.push_back(out.second);
            }
        }
    }
    OPENVINO_ASSERT(out_layouts.size() > 0, "Not found any matched output");
    return out_layouts;
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout condition_inst::calc_output_layout(condition_node const& /* node */, kernel_impl_params const& impl_param) {
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for condition_node!");
    OPENVINO_ASSERT(impl_param.get_input_layout(0).count() == 1, "layout of compare_data of condition should be {1,1,1,1}");

    OPENVINO_ASSERT(impl_param.inner_progs.size() == 2, "If(Condition) contains incorrect number of inner programs ", impl_param.inner_progs.size());
    OPENVINO_ASSERT(impl_param.io_output_maps.size() == 2, "If(Condition) contains incorrect number of io output maps ", impl_param.io_output_maps.size());

    auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
    auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);

    CLDNN_ERROR_LAYOUT_MISMATCH(impl_param.desc->id,
                                "Branch true output layout",
                                layouts_true[0],
                                "branch false output layout",
                                layouts_false[0],
                                "Layout of the branches should be the same.");

    return layouts_true[0];
}

template <class T>
static bool convert_data(memory::ptr mem, stream& stream) {
    mem_lock<T, mem_lock_type::read> lock_data{mem, stream};
    return (static_cast<float>(*lock_data.data()) != 0.f);
}

bool condition_inst::get_pred_from_memory(memory::ptr mem, stream& stream) {
    auto mem_dt = mem->get_layout().data_type;
    switch (mem_dt) {
        case cldnn::data_types::f32:
            return convert_data<float>(mem, stream);
        case cldnn::data_types::f16:
            return convert_data<ov::float16>(mem, stream);
        case cldnn::data_types::i64:
            return convert_data<int64_t>(mem, stream);
        case cldnn::data_types::i32:
            return convert_data<int32_t>(mem, stream);
        case cldnn::data_types::i8:
            return convert_data<int8_t>(mem, stream);
        case cldnn::data_types::u8:
            return convert_data<uint8_t>(mem, stream);
        case cldnn::data_types::u1:
        default:
            return convert_data<uint32_t>(mem, stream);
    }
}

static ov::PartialShape resolve_shape(const ov::PartialShape& true_pshape, const ov::PartialShape& false_pshape) {
    // true_pshape - shape of output from then_body
    // false_pshape - shape of output from else_body
    auto then_rank = true_pshape.rank();
    auto else_rank = false_pshape.rank();

    // if rangs of shapes are not equal or rang of one of them is dynamic function
    // return shape with dynamic rank
    OPENVINO_ASSERT((then_rank.is_static() && else_rank.is_static()), "dynamic rank is not supported");
    if (then_rank.get_length() != else_rank.get_length()) {
        // Union of scalar and 1D case
        if (then_rank.get_length() <= 1 && else_rank.get_length() <= 1) {
            return ov::PartialShape::dynamic(1);
        } else {
            return ov::PartialShape::dynamic();
        }
    }
    std::vector<ov::Dimension> new_dims;

    // If rangs are equal each dimesion of then_body output is union with each dimension of
    // else_body
    for (auto then_it = true_pshape.cbegin(), else_it = false_pshape.cbegin(); then_it != true_pshape.cend();
         then_it++, else_it++) {
        if ((*then_it).is_dynamic() || (*else_it).is_dynamic()) {
            new_dims.push_back(ov::Dimension::dynamic());
        } else if (*then_it == *else_it) {
            new_dims.emplace_back(*then_it);
        } else {
            auto dim_min = std::min((*then_it).get_min_length(), (*else_it).get_min_length());
            auto dim_max = std::max((*then_it).get_min_length(), (*else_it).get_min_length());
            new_dims.emplace_back(dim_min, dim_max);
        }
    }

    return ov::PartialShape(new_dims);
}

layout condition_inst::adjust_scalar_to_1d_layout(layout& target, layout& other) {
    auto target_pshape  = target.get_partial_shape();
    auto other_pshape   = other.get_partial_shape();
    auto target_rank    = target_pshape.rank();
    auto other_rank     = other_pshape.rank();
    if (target_rank.get_length() == 0 && other_rank.get_length() == 1) {
        return {ov::PartialShape{1}, target.data_type, target.format};
    }
    return target;
}

template<typename ShapeType>
std::vector<layout> condition_inst::calc_output_layouts(condition_node const& /* node */, kernel_impl_params const& impl_param) {
    if (impl_param.inner_nets.empty()) {
        OPENVINO_ASSERT(impl_param.inner_progs.empty() == false, "The count of inner programs should not be zero");
        auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
        auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_progs[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);

        const size_t num_outputs = impl_param.output_layouts.size();
        OPENVINO_ASSERT((num_outputs == layouts_true.size() && num_outputs == layouts_false.size()),
                            "The number of outputs for each branch should be same!");
        std::vector<layout> output_layouts;

        for (size_t i = 0; i < num_outputs; i++) {
            if (layouts_true[i] == layouts_false[i]) {
                output_layouts.push_back(layouts_true[i]);
            } else {
                OPENVINO_ASSERT(layouts_true[i].data_type == layouts_false[i].data_type, "data type of each branches should be same");
                OPENVINO_ASSERT(layouts_true[i].format == layouts_false[i].format, "output format of each branches should be same");
                auto out_layout = resolve_shape(layouts_true[i].get_partial_shape(), layouts_false[i].get_partial_shape());
                output_layouts.push_back(layout{out_layout, layouts_true[i].data_type, layouts_true[i].format });
            }
        }
        return output_layouts;
    } else {
        auto layouts_true  = get_output_layouts(get_out_layout_map(impl_param.inner_nets[idx_branch_true]),  impl_param.io_output_maps[idx_branch_true]);
        auto layouts_false = get_output_layouts(get_out_layout_map(impl_param.inner_nets[idx_branch_false]), impl_param.io_output_maps[idx_branch_false]);
        const size_t num_outputs = impl_param.output_layouts.size();
        OPENVINO_ASSERT((num_outputs == layouts_true.size() && num_outputs == layouts_false.size()),
                            "The number of outputs for each branch should be same!");

        auto& memory_deps = impl_param.memory_deps;
        OPENVINO_ASSERT(memory_deps.count(0) > 0, "The count of memory deps should not be zero");
        auto mem_ptr = memory_deps.at(0);
        auto pred = condition_inst::get_pred_from_memory(mem_ptr, impl_param.get_stream());
        std::vector<layout> output_layouts;
        if (pred) {
            for (size_t i = 0; i < num_outputs; i++) {
                output_layouts.push_back(condition_inst::adjust_scalar_to_1d_layout(layouts_true[i], layouts_false[i]));
            }
        } else {
            for (size_t i = 0; i < num_outputs; i++) {
                output_layouts.push_back(condition_inst::adjust_scalar_to_1d_layout(layouts_false[i], layouts_true[i]));
            }
        }
        return output_layouts;
    }
}

template std::vector<layout> condition_inst::calc_output_layouts<ov::PartialShape>(condition_node const& node, const kernel_impl_params& impl_param);

std::string condition_inst::to_string(condition_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite condition_info;

    node_info->add("condition info", condition_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
Condition primitive is reusing memory with the input.
*/
condition_inst::typed_primitive_inst(network& network, condition_node const& node)
    : parent(network, node),
      _net_true(network::allocate_network(network.get_stream_ptr(), node.get_branch_true().inner_program)),
      _net_false(network::allocate_network(network.get_stream_ptr(), node.get_branch_false().inner_program)) {
    this->set_inner_networks({_net_true, _net_false});
}

void condition_inst::update_output_layout() {
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }
        auto dep_id = _node->get_dependency(i).id();

        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
    }
    _impl_params->memory_deps = memory_deps;

    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    if (new_layouts.empty()) {
        auto new_layout = _node->type()->calc_output_layout(*_node, *_impl_params);
        new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(0), new_layout.data_padding);
        _impl_params->output_layouts[0] = new_layout;
    } else {
        for (size_t i = 0; i != new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(i), new_layout.data_padding);
            _impl_params->output_layouts[i] = new_layout;
        }
    }
}

void condition_inst::postprocess_output_memory(network::ptr executed_net, cldnn::condition::branch& branch) {
    _outputs.clear();
    _outputs.resize(outputs_memory_count());
    for (auto out_mem_map : branch.output_map) {
        auto out_mem_idx = out_mem_map.first;
        auto inner_out_id = out_mem_map.second;
        auto mem_ptr = executed_net->get_output_memory(inner_out_id);
        if (mem_ptr) {
            auto layout = _impl_params->get_output_layout(out_mem_idx);
            GPU_DEBUG_LOG << "Reshape output from " << mem_ptr->get_layout().to_short_string()
                        << " to " << layout.to_short_string() << std::endl;
            // Preallocation logic may allocate more memory than actually produced on current iteration, so we need to adjust output buffers layout
            mem_ptr = get_network().get_engine().reinterpret_buffer(*mem_ptr, layout);
        }

        _outputs[out_mem_idx] = mem_ptr;
        if (mem_ptr)
            GPU_DEBUG_LOG << "Inner net - Outputs[" << out_mem_idx << "]" << mem_ptr->get_layout().to_short_string() << std::endl;
    }
}
}  // namespace cldnn
