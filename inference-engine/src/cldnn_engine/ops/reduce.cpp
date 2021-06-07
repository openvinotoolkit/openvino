// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/reduce_logical_or.hpp"
#include "ngraph/op/reduce_logical_and.hpp"
#include "ngraph/op/reduce_l1.hpp"
#include "ngraph/op/reduce_l2.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/constant.hpp"

#include "api/reduce.hpp"
#include "api/reorder.hpp"
#include "api/reshape.hpp"

namespace CLDNNPlugin {

void CreateReduceOp(Program& p, const std::shared_ptr<ngraph::Node>& op, cldnn::reduce_mode mode, bool keep_dims) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t rank = op->get_input_shape(0).size();

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    std::vector<int32_t> rawAxes = axes_constant->cast_vector<int32_t>();

    std::vector<uint16_t> axes;
    for (size_t a = 0; a < rawAxes.size(); a++) {
        if (rawAxes[a] < 0)
            rawAxes[a] = rawAxes[a] + rank;
        if (rawAxes[a] < 0 || rawAxes[a] > rank - 1)
            IE_THROW() << op->get_friendly_name() << " Incorrect Reduce axis value: " << rawAxes[a];
        if (rank == 6) {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_w); break;
                case 3: axes.push_back(cldnn::reduce::along_z); break;
                case 4: axes.push_back(cldnn::reduce::along_y); break;
                case 5: axes.push_back(cldnn::reduce::along_x); break;
            }
        } else if (rank == 5) {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_z); break;
                case 3: axes.push_back(cldnn::reduce::along_y); break;
                case 4: axes.push_back(cldnn::reduce::along_x); break;
            }
        } else {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_y); break;
                case 3: axes.push_back(cldnn::reduce::along_x); break;
            }
        }
    }

    sort(axes.begin(), axes.end());
    axes.erase(unique(axes.begin(), axes.end()), axes.end());

    auto reducePrim = cldnn::reduce(layerName,
                                    inputPrimitives[0],
                                    mode,
                                    axes,
                                    static_cast<int32_t>(keep_dims));

    p.AddPrimitive(reducePrim);

    auto resultLayerName = layerName;
    auto out_dims = op->get_output_shape(0).size();
    if (out_dims == 3 && !keep_dims && rank >= 4) {
        resultLayerName = layerName + "_reshape";
        auto out_shape = op->get_output_shape(0);
        cldnn::tensor outTensor;
        switch (rank) {
            case 6:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]), 1, 1);
            case 5:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]), 1);
            case 4:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]));
        }
        auto reshape_prim = cldnn::reshape(resultLayerName, layerName, outTensor);
        p.AddPrimitive(reshape_prim);
        p.AddPrimitiveToProfiler(op, resultLayerName);
    }

    auto reorderLayerName = layerName + "_reorder";
    cldnn::format out_format = cldnn::format::any;
    auto out_dt = DataTypeFromPrecision(op->get_output_element_type(0));
    if (!keep_dims && rank > 4) {
        if (rank - rawAxes.size() == 6)
            out_format = cldnn::format::bfwzyx;
        else if (rank - rawAxes.size() == 5)
            out_format = cldnn::format::bfzyx;
        else if (rank - rawAxes.size() <= 4)
            out_format = cldnn::format::bfyx;

        auto reorder_prim = cldnn::reorder(reorderLayerName, resultLayerName, out_format, out_dt);
        p.AddPrimitive(reorder_prim);
        p.AddPrimitiveToProfiler(op, reorderLayerName);
    } else {
        p.AddPrimitiveToProfiler(op);
    }
}

void CreateReduceMaxOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceMax>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::max, op->get_keep_dims());
}

void CreateReduceLogicalAndOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceLogicalAnd>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::logical_and, op->get_keep_dims());
}

void CreateReduceLogicalOrOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceLogicalOr>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::logical_or, op->get_keep_dims());
}

void CreateReduceMeanOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceMean>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::mean, op->get_keep_dims());
}

void CreateReduceMinOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceMin>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::min, op->get_keep_dims());
}

void CreateReduceProdOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceProd>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::prod, op->get_keep_dims());
}

void CreateReduceSumOp(Program& p, const std::shared_ptr<ngraph::op::v1::ReduceSum>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::sum, op->get_keep_dims());
}

void CreateReduceL1Op(Program& p, const std::shared_ptr<ngraph::op::v4::ReduceL1>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::l1, op->get_keep_dims());
}

void CreateReduceL2Op(Program& p, const std::shared_ptr<ngraph::op::v4::ReduceL2>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::l2, op->get_keep_dims());
}

REGISTER_FACTORY_IMPL(v1, ReduceMax);
REGISTER_FACTORY_IMPL(v1, ReduceLogicalAnd);
REGISTER_FACTORY_IMPL(v1, ReduceLogicalOr);
REGISTER_FACTORY_IMPL(v1, ReduceMean);
REGISTER_FACTORY_IMPL(v1, ReduceMin);
REGISTER_FACTORY_IMPL(v1, ReduceProd);
REGISTER_FACTORY_IMPL(v1, ReduceSum);
REGISTER_FACTORY_IMPL(v4, ReduceL1);
REGISTER_FACTORY_IMPL(v4, ReduceL2);

}  // namespace CLDNNPlugin
