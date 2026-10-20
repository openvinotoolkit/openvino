// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/normalize.hpp"
#include "intel_gpu/primitives/data.hpp"
#include <plugin/transformations/normalize_l2_decomposition.hpp>
#include "intel_gpu/primitives/reshape.hpp"

namespace ov::intel_gpu {

static void CreateNormalizeL2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::NormalizeL2>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // params
    auto const_axis = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(const_axis != nullptr, "[GPU] Unsupported axis node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    auto axis = const_axis->cast_vector<size_t>();
    bool across_spatial = axis.size() != 1 || axis[0] != 1;
    const bool across_x = normalize_l2_supports_across_x(*op);
    float eps = op->get_eps();

    // WA for OVC outputting %.6f
    if (eps == 0.0f && !across_x) {
        eps = 1.19209e-7f;
    }

    // We create fake scale constant and fill it with ones to keep the same behavior as current primitive
    auto scale = std::make_shared<ov::op::v0::Constant>(op->get_output_element_type(0), ov::Shape{1}, std::vector<float>{1.0});
    cldnn::layout constLayout = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), cldnn::format::bfyx, cldnn::tensor{1});
    auto mem = p.get_engine().allocate_memory(constLayout, false);
    cldnn::mem_lock<int8_t> tmpPointer{mem, p.get_engine().get_service_stream()};
    auto* buf = tmpPointer.data();
    auto bufSize = scale->get_output_tensor(0).size();

    if (bufSize != constLayout.bytes_count()) {
        OPENVINO_THROW("Invalid scales buffer in NormalizeL2 op ", op->get_friendly_name());
    }

    std::memcpy(&buf[0], scale->get_data_ptr(), bufSize);
    auto scalesName = layerName + "_cldnn_input_scales";
    p.add_primitive(*op, cldnn::data(scalesName, mem));

    auto norm_input = inputs[0];
    auto norm_name = layerName;
    const bool reshape_rank3 = across_x && op->get_input_partial_shape(0).size() == 3;
    if (reshape_rank3) {
        const auto shape = op->get_input_shape(0);
        const ov::Shape expanded_shape{shape[0], shape[1], 1, shape[2]};
        const auto reshape_name = layerName + "_across_x_input";
        if (p.use_new_shape_infer()) {
            p.add_primitive(*op, cldnn::reshape(reshape_name,
                                              norm_input,
                                              false,
                                              std::vector<int64_t>(expanded_shape.begin(), expanded_shape.end()),
                                              ov::PartialShape(expanded_shape)));
        } else {
            p.add_primitive(*op, cldnn::reshape(reshape_name, norm_input, tensor_from_dims(expanded_shape)));
        }
        norm_input = cldnn::input_info(reshape_name);
        norm_name += "_across_x";
    }

    auto normPrim = cldnn::normalize(norm_name,
                                     norm_input,
                                     scalesName,
                                     across_spatial,
                                     eps,
                                     across_x);

    p.add_primitive(*op, normPrim);
    if (reshape_rank3) {
        const auto shape = op->get_output_shape(0);
        if (p.use_new_shape_infer()) {
            p.add_primitive(*op, cldnn::reshape(layerName,
                                              cldnn::input_info(norm_name),
                                              false,
                                              std::vector<int64_t>(shape.begin(), shape.end()),
                                              ov::PartialShape(shape)));
        } else {
            p.add_primitive(*op, cldnn::reshape(layerName, cldnn::input_info(norm_name), tensor_from_dims(shape)));
        }
    }
}

REGISTER_FACTORY_IMPL(v0, NormalizeL2);

}  // namespace ov::intel_gpu
