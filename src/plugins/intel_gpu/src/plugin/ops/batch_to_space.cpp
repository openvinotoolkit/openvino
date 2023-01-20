// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/batch_to_space.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/batch_to_space.hpp"

namespace ov {
namespace intel_gpu {

static void CreateBatchToSpaceOp(Program& p, const std::shared_ptr<ngraph::op::v1::BatchToSpace>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto rank = op->get_input_shape(0).size();
    auto format = cldnn::format::get_default_format(rank);

    std::vector<cldnn::tensor> tensor_inputs;
    tensor_inputs.reserve(3);

    for (size_t i = 1; i < 4; ++i) {
        auto inConst = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(i));
        if (!inConst)
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        std::vector<int32_t> sizes = inConst->cast_vector<int32_t>();
        int32_t default_size = i == 1 ? 1 : 0;
        for (size_t s = sizes.size(); s < rank; s++) {
            sizes.push_back(default_size);
        }
        tensor_inputs.emplace_back(format, sizes, default_size);
    }
    auto out_size = tensor_from_dims(op->get_output_shape(0));

    auto batchToSpacePrim = cldnn::batch_to_space(layerName,
                                                  inputs[0], // input
                                                  tensor_inputs[0], // block_shape
                                                  tensor_inputs[1], // crops_begin
                                                  tensor_inputs[2], // crops_end
                                                  out_size);

    p.add_primitive(*op, batchToSpacePrim);
}

REGISTER_FACTORY_IMPL(v1, BatchToSpace);

}  // namespace intel_gpu
}  // namespace ov
