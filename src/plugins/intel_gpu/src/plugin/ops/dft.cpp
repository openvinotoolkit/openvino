// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/common_utils.hpp>
#include <intel_gpu/primitives/dft.hpp>

#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/rdft.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void createDft(ProgramBuilder& p,
               const std::shared_ptr<ov::Node>& op,
               cldnn::dft_direction direction,
               cldnn::dft_mode mode) {
    validate_inputs_count(op, {2, 3});

    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& friendly_name = op->get_friendly_name();

    if (op->is_dynamic() && p.use_new_shape_infer()) {
        std::vector<int64_t> axes;
        auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (axes_constant != nullptr) {
            axes = axes_constant->cast_vector<int64_t>();
            uint8_t axis_correction = static_cast<uint8_t>(op->get_input_partial_shape(0).size());
            if (direction != cldnn::dft_direction::forward || mode != cldnn::dft_mode::real) {
                --axis_correction;
            }
            ov::util::try_normalize_axes(axes, axis_correction, *op);
        }

        if (op->get_input_size() == 3) {
            std::vector<int64_t> signal_size;
            auto signal_size_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (signal_size_constant != nullptr) {
                signal_size = signal_size_constant->cast_vector<int64_t>();
            }

            const cldnn::dft prim(layer_name, inputs[0], inputs[1], inputs[2], axes, signal_size, direction, mode);
            p.add_primitive(*op, prim);
        } else {
            const cldnn::dft prim(layer_name, inputs[0], inputs[1], axes, direction, mode);
            p.add_primitive(*op, prim);
        }
    } else {
        const auto& out_shape = op->get_output_shape(0);

        auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", friendly_name, " (", op->get_type_name(), ")");
        auto axes = axes_constant->cast_vector<int64_t>();
        uint8_t axis_correction = static_cast<uint8_t>(op->get_input_shape(0).size());
        if (direction != cldnn::dft_direction::forward || mode != cldnn::dft_mode::real) {
            --axis_correction;
        }
        ov::util::try_normalize_axes(axes, axis_correction, *op);

        std::vector<int64_t> signal_size;
        if (op->get_input_size() == 3) {
            auto signal_size_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            OPENVINO_ASSERT(signal_size_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", friendly_name, " (", op->get_type_name(), ")");
            signal_size = signal_size_constant->cast_vector<int64_t>();
        }

        const cldnn::dft prim(layer_name, inputs.front(), axes, signal_size, out_shape, direction, mode);

        p.add_primitive(*op, prim);
    }
}

void CreateDFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v7::DFT>& op) {
    createDft(p, op, cldnn::dft_direction::forward, cldnn::dft_mode::complex);
}

void CreateIDFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v7::IDFT>& op) {
    createDft(p, op, cldnn::dft_direction::inverse, cldnn::dft_mode::complex);
}

void CreateRDFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v9::RDFT>& op) {
    createDft(p, op, cldnn::dft_direction::forward, cldnn::dft_mode::real);
}

void CreateIRDFTOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v9::IRDFT>& op) {
    createDft(p, op, cldnn::dft_direction::inverse, cldnn::dft_mode::real);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, DFT);
REGISTER_FACTORY_IMPL(v7, IDFT);
REGISTER_FACTORY_IMPL(v9, RDFT);
REGISTER_FACTORY_IMPL(v9, IRDFT);

}  // namespace intel_gpu
}  // namespace ov
