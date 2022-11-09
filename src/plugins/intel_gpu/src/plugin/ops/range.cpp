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
    auto output_pshape = op->get_output_partial_shape(0);
    auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    std::shared_ptr<cldnn::layout> outLayout = nullptr;
    if (output_pshape.is_static()) {
        OPENVINO_ASSERT(output_pshape.rank().get_length() == 1 , "[GPU] range v4 output rank should be 1");
        auto& out_shape = op->get_output_shape(0);
        outLayout = std::make_shared<cldnn::layout>(output_dtype, cldnn::format::bfyx, cldnn::tensor(cldnn::batch(out_shape[0])));
    } else {
        outLayout = std::make_shared<cldnn::layout>(output_pshape, output_dtype, cldnn::format::bfyx);
    }

    cldnn::range prim(layer_type_name_ID(op),
                        p.GetInputPrimitiveIDs(op),
                        *outLayout);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace intel_gpu
}  // namespace ov
