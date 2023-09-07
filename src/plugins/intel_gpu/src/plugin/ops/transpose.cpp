// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTransposeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Transpose>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> order;
    if (op->get_input_size() == 2) {
        auto order_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(order_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        order = order_constant->cast_vector<uint16_t>();
    }

    auto is_convert_color_type_impl = [](const std::shared_ptr<ov::Node> &node) {
        return ov::is_type<ov::op::v8::NV12toRGB>(node) ||
               ov::is_type<ov::op::v8::NV12toBGR>(node) ||
               ov::is_type<ov::op::v8::I420toRGB>(node) ||
               ov::is_type<ov::op::v8::I420toBGR>(node);
    };

    auto is_convert_color_type = [&is_convert_color_type_impl](const std::shared_ptr<ov::Node> &node) {
        if (ngraph::is_type<ov::op::v0::Convert>(node)) {
            return is_convert_color_type_impl(node->get_input_node_shared_ptr(0));
        }
        return is_convert_color_type_impl(node);
    };

    // Handle Transpose operation related to ConvertColor operation:
    // In case of ConvertColor operation we have NHWC (byxf) input format which should be converted to
    // NCHW (bfyx) by this Permute, so we replace Permute with Reorder (to bfyx) primitve
    auto input = op->get_input_size() > 0 ? op->get_input_node_shared_ptr(0) : nullptr;
    // Handle the case ConvertColor -> FakeQuantize -> Permute
    auto input1 = input ? (input->get_input_size() > 0 ? input->get_input_node_shared_ptr(0) : nullptr) : nullptr;
    if (((input && is_convert_color_type(input)) || (input1 && is_convert_color_type(input1)))
            && order == std::vector<uint16_t>{0, 3, 1, 2}) {
        auto precision = input->get_element_type();
        auto reorder_prim = cldnn::reorder(layerName,
                                      inputs[0],
                                      cldnn::format::bfyx,
                                      cldnn::element_type_to_data_type(precision),
                                      std::vector<float>(),
                                      cldnn::reorder_mean_mode::none);
        p.add_primitive(*op, reorder_prim);
        return;
    }

    int rank = std::max(4, static_cast<int>(op->get_input_partial_shape(0).size()));
    if (order.empty()) {
        // if order size is less than 4 - fill the rest with just copy
        for (int o = rank - 1; o >= 0; o--)
            order.push_back((uint16_t)o);
    }

    auto permutePrim = cldnn::permute(layerName,
                                      inputs[0],
                                      order);
    permutePrim.output_data_types[0] = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    p.add_primitive(*op, permutePrim);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace intel_gpu
}  // namespace ov
