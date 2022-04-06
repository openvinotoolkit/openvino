// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/transpose.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateTransposeOp(Program& p, const std::shared_ptr<ngraph::op::v1::Transpose>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> order;
    if (op->get_input_size() == 2) {
        auto order_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!order_constant) {
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        order = order_constant->cast_vector<uint16_t>();
    }

    auto is_convert_color_type = [](const std::shared_ptr<ov::Node> &node) {
        return ngraph::is_type<ngraph::op::v8::NV12toRGB>(node) ||
               ngraph::is_type<ngraph::op::v8::NV12toBGR>(node) ||
               ngraph::is_type<ngraph::op::v8::I420toRGB>(node) ||
               ngraph::is_type<ngraph::op::v8::I420toBGR>(node);
    };

    // Handle Transpose operation related to ConvertColor operation:
    // In case of ConvertColor operation we have NHWC (byxf) input format which should be converted to
    // NCHW (bfyx) by this Permute, so we replace Permute with Reorder (to bfyx) primitve
    auto input = op->input(0).get_source_output().get_node_shared_ptr();
    if (is_convert_color_type(input) && order == std::vector<uint16_t>{0, 3, 1, 2}) {
        auto precision = input->get_element_type();
        p.AddPrimitive(cldnn::reorder(layerName,
                                      inputPrimitives[0],
                                      cldnn::format::bfyx,
                                      DataTypeFromPrecision(precision),
                                      std::vector<float>(),
                                      cldnn::reorder_mean_mode::none,
                                      op->get_friendly_name()));
        p.AddPrimitiveToProfiler(op);
        return;
    }

    int rank = std::max(4, static_cast<int>(op->get_input_shape(0).size()));
    if (order.empty()) {
        // if order size is less than 4 - fill the rest with just copy
        for (int o = rank - 1; o >= 0; o--)
            order.push_back((uint16_t)o);
    }

    auto permutePrim = cldnn::permute(layerName,
                                      inputPrimitives[0],
                                      order,
                                      op->get_friendly_name());

    p.AddPrimitive(permutePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
