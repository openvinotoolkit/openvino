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

void createDft(Program& p,
               const std::shared_ptr<ngraph::Node>& op,
               cldnn::dft_direction direction,
               cldnn::dft_mode mode) {
    validate_inputs_count(op, {2, 3});

    const auto inputs = p.GetInputPrimitiveIDs(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& friendly_name = op->get_friendly_name();
    const auto& out_shape = op->get_output_shape(0);

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter nodes type in " << friendly_name << " (" << op->get_type_name() << ")";
    }
    auto axes = axes_constant->cast_vector<int64_t>();
    uint8_t axis_correction = op->get_input_shape(0).size();
    if (direction != cldnn::dft_direction::forward || mode != cldnn::dft_mode::real) {
        --axis_correction;
    }
    ov::normalize_axes(op.get(), axis_correction, axes);

    std::vector<int64_t> signal_size;
    if (op->get_input_size() == 3) {
        auto signal_size_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(2));
        if (!signal_size_constant) {
            IE_THROW() << "Unsupported parameter nodes type in " << friendly_name << " (" << op->get_type_name() << ")";
        }
        signal_size = signal_size_constant->cast_vector<int64_t>();
    }

    const cldnn::dft prim(layer_name, inputs.front(), axes, signal_size, out_shape, direction, mode);

    p.add_primitive(*op, prim);
}

void CreateDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::DFT>& op) {
    createDft(p, op, cldnn::dft_direction::forward, cldnn::dft_mode::complex);
}

void CreateIDFTOp(Program& p, const std::shared_ptr<ngraph::op::v7::IDFT>& op) {
    createDft(p, op, cldnn::dft_direction::inverse, cldnn::dft_mode::complex);
}

void CreateRDFTOp(Program& p, const std::shared_ptr<ngraph::op::v9::RDFT>& op) {
    createDft(p, op, cldnn::dft_direction::forward, cldnn::dft_mode::real);
}

void CreateIRDFTOp(Program& p, const std::shared_ptr<ngraph::op::v9::IRDFT>& op) {
    createDft(p, op, cldnn::dft_direction::inverse, cldnn::dft_mode::real);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, DFT);
REGISTER_FACTORY_IMPL(v7, IDFT);
REGISTER_FACTORY_IMPL(v9, RDFT);
REGISTER_FACTORY_IMPL(v9, IRDFT);

}  // namespace intel_gpu
}  // namespace ov
