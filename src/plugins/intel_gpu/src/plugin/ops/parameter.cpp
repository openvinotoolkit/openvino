// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/concatenation.hpp"

namespace ov::intel_gpu {

static void CreateParameterOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Parameter>& op) {
    auto input_pshape = op->get_partial_shape();
    if (!p.use_new_shape_infer() && input_pshape.size() < 4) {
        input_pshape.insert(input_pshape.end(), 4 - input_pshape.size(), ov::Dimension(1));
    }

    cldnn::format input_format = cldnn::format::get_default_format(input_pshape.size());
    auto element_type = convert_to_supported_device_type(op->get_output_element_type(0));
    element_type = element_type == ov::element::boolean ? ov::element::u8 : element_type;

    // look at the expected color format of this input
    auto input_name = layer_type_name_ID(op);
    cldnn::layout input_layout(input_pshape, element_type, input_format);

    bool query_mode = p.is_query_mode();
    int64_t port_index = -1;
    if (!query_mode) {
        port_index = p.get_parameter_index(op);
        OPENVINO_ASSERT(port_index != -1, "[GPU] Parameter port index for ", input_name, " not found");
    }

    auto is_convert_color_type = [](const std::shared_ptr<ov::Node> &node) {
        return ov::is_type<ov::op::v8::NV12toRGB>(node) ||
               ov::is_type<ov::op::v8::NV12toBGR>(node) ||
               ov::is_type<ov::op::v8::I420toRGB>(node) ||
               ov::is_type<ov::op::v8::I420toBGR>(node);
    };

    std::function<bool(const std::shared_ptr<ov::Node>&, size_t)> recursive_search_convert_color =
        [&](const std::shared_ptr<ov::Node> &node, size_t curr_depth) -> bool {
        bool convert_color_found = is_convert_color_type(node);
        if (curr_depth != 0) {
            for (auto& user : node->get_users()) {
                convert_color_found |= recursive_search_convert_color(user, curr_depth - 1);
            }
        }
        return convert_color_found;
    };

    std::function<bool(const std::shared_ptr<ov::Node>&)> has_surface_input =
        [](const std::shared_ptr<ov::Node> &node) -> bool {
        bool surface_input_found = false;
        if (node->output(0).get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
            std::string mem_type = node->output(0).get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                                .as<ov::preprocess::TensorInfoMemoryType>().value;
            if (mem_type.find(ov::intel_gpu::memory_type::surface) != std::string::npos) {
                surface_input_found = true;
            }
        }
        return surface_input_found;
    };

    std::function<bool(const std::shared_ptr<ov::Node>&)> connected_to_quantize =
        [&](const std::shared_ptr<ov::Node> &node) -> bool {
        for (auto& user : node->get_users()) {
            if (ov::is_type<ov::op::v0::FakeQuantize>(user))
                return true;
        }
        return false;
    };

    size_t search_depth = 3;
    bool is_convert_color_input = recursive_search_convert_color(op, search_depth);
    bool is_surface_input = has_surface_input(op);

    if (is_surface_input) {
        size_t batch = input_pshape[0].get_length();
        input_layout.format = cldnn::format::nv12;
        input_layout.set_partial_shape({ 1, input_pshape[1], input_pshape[2], input_pshape[3] });

        if (!query_mode) {
            p.inputLayouts.insert({ port_index, input_layout });
        }

        std::string suffix = "";
        std::vector<cldnn::input_info> surfaces_inputs;
        for (size_t i = 0; i < batch; ++i) {
            if (batch > 1)
                suffix = "_" + std::to_string(i);
            std::string batched_name = input_name + suffix;
            p.add_primitive(*op, cldnn::input_layout(batched_name, input_layout));

            if (!query_mode) {
                p.inputPrimitiveIDs[port_index].emplace_back(batched_name);
            }

            auto reorder_layout = input_layout;
            reorder_layout.format = cldnn::format::bfyx;

            auto reorder_name = "reorder:" + input_name + ProgramBuilder::m_preProcessTag + suffix;
            auto reorder = cldnn::reorder(reorder_name,
                                          cldnn::input_info(batched_name),
                                          reorder_layout);
            reorder.input_mem_type = cldnn::reorder::memory_type::surface;
            p.add_primitive(*op, reorder);
            surfaces_inputs.emplace_back(reorder_name);
        }

        if (batch > 1 && !is_convert_color_input)
            p.add_primitive(*op, cldnn::concatenation(input_name, surfaces_inputs, 0));
        else
            p.primitive_ids[input_name] = "reorder:" + input_name + ProgramBuilder::m_preProcessTag;
    } else {
        auto reorder_name = "reorder:" + input_name + ProgramBuilder::m_preProcessTag;

        p.add_primitive(*op, cldnn::input_layout(input_name, input_layout));

        if (!query_mode) {
            p.inputPrimitiveIDs[port_index] = { input_name };
            p.inputLayouts.insert({ port_index, input_layout });
        }

        if (connected_to_quantize(op)) {
            // Techically this reorder is not needed, but for some reason it impacts layout propagation logic
            // TODO: Remove it and fix layout assignment & propagation passes
            p.add_primitive(*op, cldnn::reorder(reorder_name, cldnn::input_info(input_name), input_layout), {input_name});
        }
    }
}

REGISTER_FACTORY_IMPL(v0, Parameter);

}  // namespace ov::intel_gpu
