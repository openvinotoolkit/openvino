// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {

primitive_type_id reshape::type_id() {
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto prim = node.get_primitive();
    auto input_layout = node.input().get_non_padded_output_layout();

    if (input_layout.is_static() && (node.get_dependencies().size() == 2 && node.get_shape_ready())) {
        auto sizes = prim->output_shape;
        auto input_sizes = input_layout.get_dims();
        int64_t need_recalc = -1;
        ov::Dimension::value_type shape_count = 1;

        for (size_t i = 0; i < sizes.size(); i++) {
            if (sizes[i].is_dynamic()) {
                if (need_recalc >= 0) {
                    CLDNN_ERROR_MESSAGE(node.id(), "Only one dimension of the new shape can be -1");
                }
                need_recalc = i;
                continue;
            }
            shape_count *= sizes[i].get_length();
        }
        if (need_recalc >= 0)
            sizes[need_recalc] = static_cast<int>(input_layout.count()) / shape_count;

        node.reset_shape_ready();

        return layout{input_layout.data_type, input_layout.format, sizes};
    } else {
        return layout{input_layout.data_type, input_layout.format, prim->output_shape};
    }
}

std::string reshape_inst::to_string(reshape_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite reshape_info;
    reshape_info.add("input id", input.id());
    reshape_info.add("output shape", desc->output_shape);

    node_info->add("reshape info", reshape_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network& network, reshape_node const& node) : parent(network, node, false) {
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Input layout data typr",
                                    input_layout.data_type,
                                    "output layout data type",
                                    output_layout.data_type,
                                    "");
    // if (node.get_dependencies().size() == 1) {
    //     CLDNN_ERROR_NOT_EQUAL(node.id(),
    //                         "Output layout count",
    //                         output_layout.count(),
    //                         "input layout count",
    //                         input_layout.count(),
    //                         "Output layout of reshape primitive changes size of input buffer");
    // }

    // if reshape operated in-place, postpone creation of the output until network run,
    // then create new memory object as the reinterpreted output of the previous primitive
    if (!node.can_be_optimized() && _node.get_output_layout().is_static())
        _output = allocate_output();
    else if (_exec_deps.size() > 0 && input_memory_ptr())
        reuse_input();
}

static std::vector<int64_t> read_vector(cldnn::memory::ptr mem, cldnn::stream& stream) {
    switch (mem->get_layout().data_type) {
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        default: IE_THROW() << "read_vector: unsupported data type";
    }
}

void reshape_inst::update_shape() {
    auto& node = const_cast<reshape_node&>(dynamic_cast<const reshape_node&>(_node));
    if (_node.get_dependencies().size() == 2) {
        auto shape_mem = _network.get_output_memory(_node.get_dependency(1).id());
        // TODO: usm_device is copied to host on lock(), but we need to ensure that this is better, then
        // keeping such constants on host (i.e. modifying transfer_memory_to_device)
        // if (shape_mem->get_allocation_type() == allocation_type::usm_device) {
        //     IE_THROW() << " lockable memory is required to update shape for reshape prim\n";
        // }
        auto reshape_prim = std::static_pointer_cast<reshape>(std::const_pointer_cast<primitive>(_node.get_primitive()));
        reshape_prim->output_shape = ov::PartialShape(read_vector(shape_mem, _network.get_stream()));
        node.set_shape_ready();
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto new_layout = _node.type()->calc_output_layout(_node);
    auto out_layout = _node.is_valid_output_layout() ? _node.get_output_layout() : layout(data_types::f32, format::any, tensor{});
    auto out_layout_str = _node.is_valid_output_layout() ? out_layout.to_string() : "invalid";
    GPU_DEBUG_IF(debug_config->verbose >= 4) {
        GPU_DEBUG_COUT << id() << " update shape: was: " << out_layout_str << " now: " << new_layout.to_string() << std::endl;
    }
    if (!_node.is_valid_output_layout() || _node.get_output_layout() != new_layout)
        set_shape_change();
    // TODO: Get rid of this const_cast
    node.set_output_layout(new_layout);
}

void reshape_inst::on_execute() {
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input() {
    build_deps();  // reshape need deps

    if (!input_memory_ptr())
        throw std::runtime_error("[GPU] Reshape can't reuse nullptr input memory");

    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}

}  // namespace cldnn
