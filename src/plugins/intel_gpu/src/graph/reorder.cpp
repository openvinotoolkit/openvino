// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "reorder_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif // ENABLE_ONEDNN_FOR_GPU
#include <algorithm>
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(reorder)

template<typename ShapeType>
std::vector<layout> reorder_inst::calc_output_layouts(reorder_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<reorder>();
    auto input_layout = impl_param.get_input_layout();

    auto ifmt = input_layout.format;
    auto ofmt = desc->output_format == format::any ? ifmt : desc->output_format;

    if (desc->weights_reorder_params) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(desc->weights_reorder_params);
        if (onednn_weights_params && input_layout.format != onednn::find_data_format(onednn_weights_params->_in_desc)) {
            onednn_weights_params->_in_desc = onednn::layout_to_memory_desc(input_layout);
        }
#endif // ENABLE_ONEDNN_FOR_GPU
        return { desc->weights_reorder_params->get_output_layout() };
    } else {
        return { layout(input_layout.get<ShapeType>(), desc->output_data_types[0].value(), ofmt, desc->output_paddings[0]) };
    }
}

std::string reorder_inst::to_string(reorder_node const& node) {
    auto desc = node.get_primitive();
    auto mean = desc->mean;
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    auto input_mem_type = desc->input_mem_type ==
        reorder::memory_type::buffer ? "buffer" : "surface";

    json_composite reorder_info;
    reorder_info.add("input id", input.id());
    reorder_info.add("mean", mean);
    reorder_info.add("input mem type", input_mem_type);
    if (desc->subtract_per_feature.size() > 0) {
        reorder_info.add("subtract per feature", desc->subtract_per_feature);
    }

    node_info->add("reorder info", reorder_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reorder_inst::typed_primitive_inst(network& network) : parent(network) {
    _type = reorder::type_id();
}

reorder_inst::typed_primitive_inst(network& network, reorder_node const& node) :
        parent(network, node, !node.can_be_optimized()
                              && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound()))
        , _req_reinterpr(node.requires_reinterpret()) {
    update_output_memory();

    if (is_dynamic())
        return;

    auto input_layout = node.get_input_layout();
    auto output_layout = node.get_output_layout();
    if (input_layout.is_static() && output_layout.is_static()) {
        CLDNN_ERROR_LESS_THAN(node.id(),
                              "Input dimension size",
                              input_layout.get_tensor().raw.size(),
                              "ouput dimension size",
                              output_layout.get_tensor().raw.size(),
                              "Input dimension < output dimension. Reorder primitive woks only with same dimension sizes "
                              "(reorder) or when input > output (flatten).");
    }
    if (!argument->subtract_per_feature.empty()) {
        CLDNN_ERROR_GREATER_THAN(node.id(),
                                 "Input feature dimension size",
                                 input_layout.get_tensor().feature.size(),
                                 "value",
                                 1,
                                 "Subtracting values work only for formats that have feature dimension == 1");
        if (input_layout.format != format::nv12) {
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                "Input feature size[0]",
                static_cast<size_t>(input_layout.feature()),
                "argument subtract per feature size",
                argument->subtract_per_feature.size(),
                "Number of features/channels in input does not match the number of features/channels in "
                "values to subtract");
        }
    }
}

void reorder_inst::on_execute() {
    update_output_memory();
}

void reorder_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory())
        && output_memory().get_layout().identical(get_output_layout()))
        return;

    if (_node != nullptr)
        build_deps();

    // Do not update output memory when reorder is optimized out
    // but input memory is not allocated yet because input is dynamic.
    // Since dep's _outputs may be empty, Check whether input memory is null by dep's outputs_allocated()
    if (!dependencies().front().first->outputs_allocated())
        return;

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_property(ov::intel_gpu::enable_memory_pool)) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }

    if (requires_reinterpret()) {
        _outputs[0] = _network.get_engine().reinterpret_buffer(input_memory(), get_output_layout());
    } else {
        _outputs[0] = input_memory_ptr();
    }
    _mem_allocated = false;
}
}  // namespace cldnn
