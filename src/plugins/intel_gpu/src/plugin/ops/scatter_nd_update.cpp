// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/scatter_nd_update.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/scatter_nd_update.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateScatterNDUpdateOp(Program& p, const std::shared_ptr<ngraph::op::v3::ScatterNDUpdate>& op) {
    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto indices_rank = op->get_input_shape(1).size();

    auto indices_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (indices_constant) {
        auto indices = indices_constant->cast_vector<int32_t>();
        auto indices_last_dim = op->get_input_shape(1)[indices_rank - 1];
        auto data_shape = op->get_input_shape(0);
        bool valid = true;
        for (int i = 0; i < indices.size(); ++i) {
            if (indices[i] >= data_shape[i % indices_last_dim])
                valid = false;
        }

        if (!valid)
           IE_THROW() << "Invaild indices values";
    }

    auto primitive = cldnn::scatter_nd_update(layerName,
                                              inputPrimitives[0],
                                              inputPrimitives[1],
                                              inputPrimitives[2],
                                              indices_rank,
                                              op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, ScatterNDUpdate);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
