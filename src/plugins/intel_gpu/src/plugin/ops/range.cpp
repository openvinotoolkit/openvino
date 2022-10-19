// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/program.hpp>
#include <intel_gpu/plugin/common_utils.hpp>

#include <intel_gpu/primitives/range.hpp>
#include <ngraph/op/range.hpp>

namespace ov {
namespace intel_gpu {

static void CreateRangeOp(Program &p, const std::shared_ptr<ngraph::op::v4::Range> &op) {
    validate_inputs_count(op, { 3 });

    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto outDataType = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    auto outRank = op->get_output_partial_shape(0).rank();
    if (outRank.is_static()) {
        OPENVINO_ASSERT(outRank.get_length() == 1, "[GPU] Range v4 output rank is not equal to 1");
    }

    if (p.use_new_shape_infer()) {
        cldnn::range prim(layerName, inputPrimitives, outDataType);
        p.add_primitive(*op, prim);
    } else {
        auto &outShape = op->get_output_shape(0);
        cldnn::tensor outTensor(cldnn::spatial(outShape[0]));
        cldnn::range prim(layerName, inputPrimitives, outTensor, outDataType);
        p.add_primitive(*op, prim);
    }
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace intel_gpu
}  // namespace ov
