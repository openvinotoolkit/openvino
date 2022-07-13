// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/plugin/program.hpp>
#include <intel_gpu/primitives/dft.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/dft.hpp>

namespace ov {
namespace intel_gpu {

namespace {

void createDft(Program& p, const std::shared_ptr<ngraph::Node>& op, cldnn::dft_kind kind) {
    p.ValidateInputs(op, {2, 3});

    const auto inputs = p.GetInputPrimitiveIDs(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& op_friendly_name = op->get_friendly_name();
    const auto& out_shape = op->get_output_shape(0);

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter nodes type in " << op_friendly_name << " (" << op->get_type_name() << ")";
    }
    auto axes = axes_constant->cast_vector<int64_t>();
    const uint8_t data_rank = out_shape.size();
    ov::normalize_axes(op.get(), data_rank - 1, axes);

    const cldnn::dft prim(layer_name, inputs.front(), std::move(axes), out_shape, kind, op_friendly_name);

    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op);
}

void CreateDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::DFT>& op) {
    createDft(p, op, cldnn::dft_kind::forward);
}

void CreateIDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::IDFT>& op) {
    createDft(p, op, cldnn::dft_kind::inverse);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, DFT);
REGISTER_FACTORY_IMPL(v7, IDFT);

}  // namespace intel_gpu
}  // namespace ov
