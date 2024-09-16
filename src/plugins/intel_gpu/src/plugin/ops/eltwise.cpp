// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/eltwise.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/xor.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

void CreateElementwiseOp(ProgramBuilder& p,
                         const std::shared_ptr<ov::Node>& op,
                         cldnn::eltwise_mode mode,
                         std::vector<float> coefficients,
                         bool pythondiv) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto out_pshape = op->get_output_partial_shape(0);
    auto out_rank = out_pshape.size();
    // New shape infer is supposed to work w/o extra reshapes/reorders
    // So the code below must be removed once new shape infer is enabled
    if (out_pshape.is_static() && !p.use_new_shape_infer()) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input_pshape = op->get_input_partial_shape(i);
            auto input_rank = input_pshape.size();
            if (input_rank != out_rank && input_pshape.is_static()) {
                // Add reorder if changing number of dimensions requires changing format
                auto targetFormat = cldnn::format::get_default_format(out_rank);
                if (targetFormat.value != cldnn::format::get_default_format(input_rank).value) {
                    auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                    auto targetDatatype = cldnn::element_type_to_data_type(op->get_input_element_type(i));
                    auto reorderPrim = cldnn::reorder(reorderName,
                                                    inputs[i],
                                                    targetFormat,
                                                    targetDatatype);

                    p.add_primitive(*op, reorderPrim);
                    inputs[i] = cldnn::input_info(reorderName);
                }

                auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

                // Extend input dimensions by prepending ones
                input_pshape.insert(input_pshape.begin(), out_rank - input_rank, 1ul);

                auto targetShape = tensor_from_dims(input_pshape.to_shape());

                auto reshapePrim = cldnn::reshape(reshapeName, inputs[i], targetShape);
                p.add_primitive(*op, reshapePrim);

                inputs[i] = cldnn::input_info(reshapeName);
            }
        }
    }

    auto out_dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    auto eltwisePrim = cldnn::eltwise(layerName,
                                      inputs,
                                      mode,
                                      std::move(coefficients),
                                      out_dt,
                                      op->get_autob(),
                                      pythondiv);

    p.add_primitive(*op, eltwisePrim);
}

static void CreateAddOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Add>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sum);
}

static void CreateMultiplyOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Multiply>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::prod);
}

static void CreateMaximumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Maximum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::max);
}

static void CreateMinimumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Minimum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::min);
}

static void CreateSubtractOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Subtract>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sub);
}

static void CreateDivideOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Divide>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::div, {}, op->is_pythondiv());
}

static void CreateSquaredDifferenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::SquaredDifference>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::squared_diff);
}

static void CreateEqualOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Equal>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::eq);
}

static void CreateNotEqualOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::NotEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ne);
}

static void CreateLessOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Less>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::lt);
}

static void CreateLessEqualOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::LessEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::le);
}

static void CreateGreaterOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Greater>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::gt);
}

static void CreateGreaterEqualOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::GreaterEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ge);
}

static void CreateLogicalAndOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::LogicalAnd>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_and);
}

static void CreateLogicalOrOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::LogicalOr>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_or);
}

static void CreateLogicalXorOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::LogicalXor>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_xor);
}

static void CreatePowerOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Power>& op) {
    validate_inputs_count(op, {2});
    auto power_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (power_node) {
        if (ov::shape_size(power_node->get_output_shape(0)) == 1) {
            float pow;
            if (!ov::op::util::get_single_value(power_node, pow))
                OPENVINO_THROW("Invalid parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            CreateUnaryEltwiseOp(p, op, cldnn::activation_func::pow, {pow});
            return;
        }
    }
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::pow);
}

static void CreateFloorModOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::FloorMod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::floor_mod);
}

static void CreateModOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Mod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::mod);
}

static void CreateIsFiniteOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v10::IsFinite>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_finite);
}

static void CreateIsInfOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v10::IsInf>& op) {
    const auto& attributes = op->get_attributes();
    const auto detect_negative = static_cast<float>(attributes.detect_negative);
    const auto detect_positive = static_cast<float>(attributes.detect_positive);
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_inf, {detect_negative, detect_positive});
}

static void CreateIsNaNOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v10::IsNaN>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_nan);
}

static void CreateBitwiseRightShiftOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::BitwiseRightShift>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::right_shift);
}

static void CreateBitwiseLeftShiftOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::BitwiseLeftShift>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::left_shift);
}

static void CreateBitwiseAndOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::BitwiseAnd>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::bitwise_and);
}

static void CreateBitwiseOrOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::BitwiseOr>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::bitwise_or);
}

static void CreateBitwiseXorOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::BitwiseXor>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::bitwise_xor);
}

REGISTER_FACTORY_IMPL(v1, Add);
REGISTER_FACTORY_IMPL(v1, Multiply);
REGISTER_FACTORY_IMPL(v1, Maximum);
REGISTER_FACTORY_IMPL(v1, Minimum);
REGISTER_FACTORY_IMPL(v1, Subtract);
REGISTER_FACTORY_IMPL(v1, Divide);
REGISTER_FACTORY_IMPL(v0, SquaredDifference);
REGISTER_FACTORY_IMPL(v1, Equal);
REGISTER_FACTORY_IMPL(v1, NotEqual);
REGISTER_FACTORY_IMPL(v1, Less);
REGISTER_FACTORY_IMPL(v1, LessEqual);
REGISTER_FACTORY_IMPL(v1, Greater);
REGISTER_FACTORY_IMPL(v1, GreaterEqual);
REGISTER_FACTORY_IMPL(v1, LogicalAnd);
REGISTER_FACTORY_IMPL(v1, LogicalOr);
REGISTER_FACTORY_IMPL(v1, LogicalXor);
REGISTER_FACTORY_IMPL(v1, Power);
REGISTER_FACTORY_IMPL(v1, FloorMod);
REGISTER_FACTORY_IMPL(v1, Mod);
REGISTER_FACTORY_IMPL(v10, IsFinite);
REGISTER_FACTORY_IMPL(v10, IsInf);
REGISTER_FACTORY_IMPL(v10, IsNaN);
REGISTER_FACTORY_IMPL(v13, BitwiseAnd);
REGISTER_FACTORY_IMPL(v13, BitwiseOr);
REGISTER_FACTORY_IMPL(v13, BitwiseXor);
REGISTER_FACTORY_IMPL(v15, BitwiseRightShift);
REGISTER_FACTORY_IMPL(v15, BitwiseLeftShift);

}  // namespace intel_gpu
}  // namespace ov
