// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id fully_connected::type_id() {
    static primitive_type_base<fully_connected> instance;
    return &instance;
}

namespace {
bool is_batch_after_spatial(const std::string order) {
    bool spatial_found = false;
    for (auto c : order) {
        switch (c) {
            case 'b':
            case 'n':
                return spatial_found;

            case 'x':
            case 'y':
            case 'z':
            case 'w':
            case 's':
                spatial_found = true;
                break;

            default:
                break;
        }
    }
    return false;
}

format::type get_preferred_format(const fully_connected_node& node) {
    auto input_layout = node.input().get_output_layout();

    if (input_layout.is_dynamic())
        return format::bfyx;

    // for 3d output we have to chose bfyx format
    if (node.get_primitive()->input_size == 3)
        return format::bfyx;

    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        (is_batch_after_spatial(input_layout.format.order()) ||
         input_layout.format == format::bs_x_bsv16 ||
         input_layout.format == format::bs_xs_xsv8_bsv8))
        return format::yxfb;

    bool no_spatial_padding = true;
    // C++ 11 range loop shouldn't be used here because of incorrect iterator functionality in mutable_array_ref<>
    for (size_t i = 0; i < input_layout.data_padding.lower_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.lower_size().spatial[i] == 0);
    }
    for (size_t i = 0; i < input_layout.data_padding.upper_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.upper_size().spatial[i] == 0);
    }

    if (input_layout.data_type == data_types::f32 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_layout.batch() != 8)
        return format::bfyx;

    auto input_pitches = input_layout.get_pitches();
    if (input_layout.data_type == data_types::f16 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_pitches.batch[0] % 2 == 0 &&
        input_layout.batch() != 16)
        return format::bfyx;

    // this condition tests whether our input is batch>1 in bfyx format, if yes there will be
    // extra reorder between input and this fc from bfyx to yxfb format (so
    // "is_batch_after_spatial" should return true)
    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        input_layout.format == format::bfyx &&
        input_layout.batch() > 1)
        return format::yxfb;

    return format::bfyx;
}

}  // namespace

layout fully_connected_inst::calc_output_layout(fully_connected_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();
    auto output_type = input_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && desc->output_data_type)
        output_type = *desc->output_data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    format output_format = get_preferred_format(node);
    if (input_layout.is_dynamic()) {
        auto batch = input_layout.size[0];
        auto feature = input_layout.size[1];
        auto output_size = ov::PartialShape{batch, weights_layout.batch(), 1, 1};
        if (desc->input_size == 3) {
            output_size = ov::PartialShape{batch, feature, 1, weights_layout.batch()};
        }

        return layout(output_type, output_format, output_size);
    } else {
        auto output_size = tensor(input_layout.batch(), weights_layout.batch(), 1, 1);
        if (desc->input_size == 3) {
            output_size = tensor(input_layout.batch(), input_layout.feature(), 1, weights_layout.batch());
        }

        return layout(output_type, output_format, output_size);
    }
}

std::string fully_connected_inst::to_string(fully_connected_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto weights_id = desc->weights;

    std::stringstream primitive_description;

    json_composite fc_info;
    fc_info.add("weights id", weights_id);
    fc_info.add("bias id", bias_id);

    node_info->add("fully connected info", fc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network& network, fully_connected_node const& node)
    : parent(network, node) { }

// TODO: looks like we can move this logic to primitive_inst class
// and call that based in some virtual method result which would return bool
// I.e.
// virual bool is_weightable_layer() { return false; }
// override this for fc/conv/etc
// and in common primitive_execute we call
// if (is_weightable_layer()) update_weights();
void fully_connected_inst::update_weights() {
    if (!_impl)
        return;


    auto& weights_params = _impl->_weights_reorder_params;
    layout expected_layout = from_weights_tensor(weights_params.dest);
    layout current_layout = node.get_dependency(1).get_output_layout();

    bool requires_reorder = weights_params.engine != kernel_selector::GenericKernelParams::Engine::NONE &&
                            (!reordered_weights || reordered_weights->get_layout() != expected_layout);
    if (requires_reorder) {
        auto& program = _node.get_program();
        auto& engine = _network.get_engine();
        auto& stream = _network.get_stream();
        auto _kernel_id = program.add_kernel(weights_params.clKernel->code.kernelString);
        program.compile();
        auto kernel = program.get_kernel(_kernel_id);

        reordered_weights = engine.allocate_memory(expected_layout, allocation_type::usm_device);

        kernel_arguments_data args;
        args.inputs.push_back(dep_memory_ptr(1));
        args.output = reordered_weights;
        stream.set_arguments(*kernel, weights_params.clKernel->params, args);
        auto out_ev = stream.enqueue_kernel(*kernel, weights_params.clKernel->params, args, {}, true);
        stream.wait_for_events({out_ev});
    }
}

}  // namespace cldnn
