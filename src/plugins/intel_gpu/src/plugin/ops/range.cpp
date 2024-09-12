// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/primitives/range.hpp>

namespace ov {
namespace intel_gpu {

static void CreateRangeOp(ProgramBuilder &p, const std::shared_ptr<ov::op::v4::Range> &op) {
    validate_inputs_count(op, { 3 });
    auto output_pshape = op->get_output_partial_shape(0);
    OPENVINO_ASSERT(output_pshape.rank().get_length() == 1 , "[GPU] range v4 output rank should be 1");
    auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    auto range_prim = std::make_shared<cldnn::range>(layer_type_name_ID(op),
                                                     p.GetInputInfo(op),
                                                     output_dtype);

    p.add_primitive(*op, range_prim);
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace intel_gpu
}  // namespace ov
