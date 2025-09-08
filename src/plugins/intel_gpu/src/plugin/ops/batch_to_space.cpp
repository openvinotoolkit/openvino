// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/batch_to_space.hpp"

namespace ov::intel_gpu {

static void CreateBatchToSpaceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::BatchToSpace>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto output_pshape = op->get_output_partial_shape(0);
    auto out_size = output_pshape.is_static() ? tensor_from_dims(output_pshape.to_shape()) : cldnn::tensor();

    bool constant_shape = true;
    for (size_t i = 1; i < 4; ++i) {
        auto inConst = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));
        if (!inConst) {
            constant_shape = false;
            break;
        }
    }

    if (!p.use_new_shape_infer() && !op->is_dynamic() && constant_shape) {
        std::vector<cldnn::tensor> tensor_inputs;
        auto rank = op->get_input_partial_shape(0).size();
        auto format = cldnn::format::get_default_format(rank);
        for (size_t i = 1; i < 4; ++i) {
            auto inConst = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));

            std::vector<int64_t> sizes = inConst->cast_vector<ov::Dimension::value_type>();
            int64_t default_size = i == 1 ? 1 : 0;
            for (size_t s = sizes.size(); s < format.dimension(); s++) {
                sizes.push_back(default_size);
            }
            tensor_inputs.emplace_back(format, sizes, default_size);
        }

        auto batchToSpacePrim = cldnn::batch_to_space(layerName,
                                                      inputs[0],            // input
                                                      tensor_inputs[0],     // block_shape
                                                      tensor_inputs[1],     // crops_begin
                                                      tensor_inputs[2],     // crops_end
                                                      out_size);

        p.add_primitive(*op, batchToSpacePrim);
    } else {
        auto batchToSpacePrim = cldnn::batch_to_space(layerName, inputs, out_size);
        p.add_primitive(*op, batchToSpacePrim);
    }
}

REGISTER_FACTORY_IMPL(v1, BatchToSpace);

}  // namespace ov::intel_gpu
