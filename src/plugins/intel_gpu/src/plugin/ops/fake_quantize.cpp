// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/fake_quantize.hpp"

#include "intel_gpu/primitives/quantize.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateFakeQuantizeOp(Program& p, const std::shared_ptr<ngraph::op::v0::FakeQuantize>& op) {
    p.ValidateInputs(op, {5});
    std::string layerName = layer_type_name_ID(op);
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    auto input_id       = inputPrimitives[0];
    auto input_low_id   = inputPrimitives[1];
    auto input_high_id  = inputPrimitives[2];
    auto output_low_id  = inputPrimitives[3];
    auto output_high_id = inputPrimitives[4];

    int levels = static_cast<int>(op->get_levels());
    auto dt = DataTypeFromPrecision(op->get_output_element_type(0));
    auto quantizationPrim = cldnn::quantize(layerName,
                                            input_id,
                                            input_low_id,
                                            input_high_id,
                                            output_low_id,
                                            output_high_id,
                                            levels,
                                            dt,
                                            op->get_friendly_name());

    p.AddPrimitive(quantizationPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, FakeQuantize);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
