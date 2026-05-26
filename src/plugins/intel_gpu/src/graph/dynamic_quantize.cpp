// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "dynamic_quantize_inst.h"
#include "fully_connected_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <limits>
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_quantize);

static bool can_skip_for_fully_connected(dynamic_quantize_node const& node, fully_connected_node const& fc_node) {
    const auto& attrs = node.get_primitive()->attrs;

    if (attrs.quantization_type != ov::op::internal::DynamicQuantize::QuantizationType::Symmetric ||
        attrs.quantization_dt != ov::element::i8 ||
        attrs.scale_dt != ov::element::f16 ||
        attrs.precomputed_reduction ||
        attrs.group_sizes.empty() ||
        attrs.group_sizes.back() == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    bool has_equivalent_fc_fast_path = false;
    const auto& forced_impls = fc_node.get_program().get_config().get_force_implementations();
    auto forced_impl = forced_impls.find(fc_node.id());
    if (forced_impl != forced_impls.end()) {
        if (forced_impl->second.impl_type != impl_types::ocl) {
            return false;
        }
        if (!forced_impl->second.kernel_name.empty() &&
            forced_impl->second.kernel_name.find("fully_connected_gpu_bf_tiled") == std::string::npos) {
            return false;
        }
        has_equivalent_fc_fast_path = forced_impl->second.kernel_name.find("fully_connected_gpu_bf_tiled") != std::string::npos;
    }

    const auto* fc_impl = fc_node.get_selected_impl();
    if (fc_impl != nullptr) {
        if (fc_impl->is_onednn()) {
            return false;
        }
        has_equivalent_fc_fast_path |= fc_impl->get_kernel_name().find("fully_connected_gpu_bf_tiled") != std::string::npos;
    }

    if (fc_node.get_preferred_impl_type() == impl_types::onednn && !has_equivalent_fc_fast_path) {
        return false;
    }

    return has_equivalent_fc_fast_path;
}

// We should skip dynamic_quantization execution for 2nd token of LLM because it does not show performance gain.
// can_be_optimized flag will be turned on from primitive_inst::update_shape function
static bool should_skip_execution(dynamic_quantize_node const& node, const layout &act_layout) {
    if (!node.is_runtime_skippable()
        || !act_layout.is_static())
        return false;

    // Do not skip dynamic quantization if next node is not fully connected.(such as SDPA)
    OPENVINO_ASSERT(node.get_users().size() == node.get_outputs_count(),
                    "Dynamic quantization is supposed to have only one user-node with duplicated connection: ", node.id());
    if (!(*node.get_users().begin())->is_type<fully_connected>())
        return false;

    auto& fc_user = (*node.get_users().begin())->as<fully_connected>();
    if (!can_skip_for_fully_connected(node, fc_user)) {
        GPU_DEBUG_TRACE << node.id() << "  dyn_quan is not runtime-skipped: no equivalent FC fast path" << std::endl;
        return false;
    }

    size_t input_batch = act_layout.batch();
    if (act_layout.format == format::bfyx && act_layout.get_partial_shape().size() != 2) {
        // 3D input
        input_batch = act_layout.batch() * act_layout.feature();
    }

    const auto dynamic_quantization_threshold = node.get_program().get_config().get_dynamic_quantization_threshold();
    if (dynamic_quantization_threshold != 0 && dynamic_quantization_threshold >= input_batch) {
        GPU_DEBUG_TRACE << node.id() << "  dyn_quan is turned off: input batch size is too small - " << input_batch << " / "
                        << dynamic_quantization_threshold << std::endl;
        return true;
    }

    return false;
}

layout dynamic_quantize_inst::calc_output_layout(dynamic_quantize_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::__calc_output_layouts(const dynamic_quantize_node &node,
                                                                 const layout &act_layout,
                                                                 const dynamic_quantize::Attributes& attrs) {
    ov::op::internal::DynamicQuantize op;
    op.set_attrs(attrs);

    auto output_format = act_layout.format;

    std::vector<ShapeType> input_shapes = {
        act_layout.get<ShapeType>(),
    };

    auto output_shapes = ov::op::internal::DynamicQuantize::shape_infer(&op, input_shapes);

    std::vector<layout> output_layouts = {  layout(output_shapes[0], attrs.quantization_dt, output_format),
                                            layout(output_shapes[1], attrs.scale_dt, output_format) };

    auto flag_skip_execution = should_skip_execution(node, act_layout);

    GPU_DEBUG_TRACE_DETAIL << node.id() << "  should_skip_execution " << flag_skip_execution << std::endl;

    if (attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        output_layouts.emplace_back(layout(output_shapes[2], attrs.zp_dt, output_format));
    }
    if (attrs.precomputed_reduction) {
        OPENVINO_ASSERT(output_layouts.size() == 2,
                        "Dynamic quantization is supposed to have 2 outputs in front of precomputed reduction, but got: ", output_layouts.size());
        output_layouts.emplace_back(layout(output_shapes[2], attrs.precomputed_reduction_dt, output_format));
    }

    if (flag_skip_execution) {
        // When execution is skipped, output data type is same as input data type
        output_layouts[0] = act_layout;

        // size of other dyn_quan outputs should be 0
        for (size_t i = 1; i < output_layouts.size(); i++) {
            *output_shapes[i].begin() = 0;
            output_layouts[i].set_partial_shape(output_shapes[i]);
        }
    }

    return output_layouts;
}

template std::vector<layout> dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(const dynamic_quantize_node &node,
                                                                                            const layout &act_layout,
                                                                                            const dynamic_quantize::Attributes& config);

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();

    return __calc_output_layouts<ov::PartialShape>(node, input_layout, desc->attrs);
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(const dynamic_quantize_node &node,
                                                                                          const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite dynamic_quantize_info;
    dynamic_quantize_info.add("output_storage_type", static_cast<int>(desc->attrs.output_storage_type));
    dynamic_quantize_info.add("scales_zp_output_order", desc->attrs.scales_zp_output_order);
    dynamic_quantize_info.add("group_sizes", desc->attrs.group_sizes);
    dynamic_quantize_info.add("quantization_dt", desc->attrs.quantization_dt);
    dynamic_quantize_info.add("scale_dt", desc->attrs.scale_dt);
    dynamic_quantize_info.add("zp_dt", desc->attrs.zp_dt);
    dynamic_quantize_info.add("quantization_type", static_cast<int>(desc->attrs.quantization_type));
    dynamic_quantize_info.add("precomputed_reduction", desc->attrs.precomputed_reduction);
    node_info->add("dynamic_quantize info", dynamic_quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

void dynamic_quantize_inst::on_execute() {
    update_output_memory();
}

void dynamic_quantize_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (_node != nullptr)
        build_deps();

    if (input_memory_ptr() == nullptr)
        return;

    if (static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory())
        && output_memory().get_layout().identical(get_output_layout()))
        return;

    OPENVINO_ASSERT(input_memory_ptr() != nullptr, "[GPU] Failed to reuse input in ", id(), " primitive: input memory was not allocated");

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        get_node().get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), get_node().get_unique_id(), get_node().id(), _network.get_id());
    }

    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}
}  // namespace cldnn
