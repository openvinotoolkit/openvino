// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/ctc_loss.hpp"

#include <ngraph/op/ctc_loss.hpp>

#include "intel_gpu/plugin/program.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateCTCLossOp(Program& p, const std::shared_ptr<ngraph::op::v4::CTCLoss>& op) {
    validate_inputs_count(op, {4, 5});

    const cldnn::ctc_loss ctc_loss_prim(layer_type_name_ID(op),
                                        p.GetInputInfo(op),
                                        op->get_preprocess_collapse_repeated(),
                                        op->get_ctc_merge_repeated(),
                                        op->get_unique());

    p.add_primitive(*op, ctc_loss_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v4, CTCLoss);

}  // namespace intel_gpu
}  // namespace ov
