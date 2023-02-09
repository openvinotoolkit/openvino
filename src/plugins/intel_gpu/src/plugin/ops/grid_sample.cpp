// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/grid_sample.hpp"

#include "intel_gpu/plugin/program.hpp"
#include "ngraph/op/grid_sample.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateGridSampleOp(Program& p, const std::shared_ptr<ngraph::op::v9::GridSample>& op) {
    validate_inputs_count(op, {2});

    const cldnn::grid_sample grid_sample_prim(layer_type_name_ID(op), p.GetInputInfo(op), op->get_attributes());

    p.add_primitive(*op, grid_sample_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v9, GridSample);

}  // namespace intel_gpu
}  // namespace ov
