// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/primitives/range.hpp>

namespace ov::intel_gpu {

static void CreateRangeOp(ProgramBuilder &p, const std::shared_ptr<ov::op::v4::Range> &op) {
    validate_inputs_count(op, { 3 });
    auto output_pshape = op->get_output_partial_shape(0);
    OPENVINO_ASSERT(output_pshape.rank().get_length() == 1 , "[GPU] range v4 output rank should be 1");
    auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    std::shared_ptr<cldnn::range> range_prim = nullptr;
    if (p.use_new_shape_infer()) {
        range_prim = std::make_shared<cldnn::range>(layer_type_name_ID(op),
                                                    p.GetInputInfo(op),
                                                    output_dtype);
    } else {
        auto outLayout = cldnn::layout{ output_pshape, output_dtype, cldnn::format::bfyx };
        range_prim = std::make_shared<cldnn::range>(layer_type_name_ID(op),
                                                    p.GetInputInfo(op),
                                                    outLayout);
    }
    p.add_primitive(*op, range_prim);
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace ov::intel_gpu
