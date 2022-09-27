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
    auto &outShape = op->get_output_shape(0);
    {
        auto r = outShape.size();
        if (r != 1)
            throw std::runtime_error { "range v4 output rank is " + std::to_string(r) };
    }
    cldnn::tensor outTensor { cldnn::spatial(outShape[0]) };
    auto outDataType = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    cldnn::layout outLayout { outDataType, cldnn::format::bfyx, outTensor };
    cldnn::range prim { layer_type_name_ID(op), p.GetInputPrimitiveIDs(op), outLayout };
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace intel_gpu
}  // namespace ov
