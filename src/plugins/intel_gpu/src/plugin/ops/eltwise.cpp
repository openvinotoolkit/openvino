// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/squared_difference.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/floor_mod.hpp"

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

void CreateElementwiseOp(Program& p,
                         const std::shared_ptr<ngraph::Node>& op,
                         cldnn::eltwise_mode mode,
                         std::vector<float> coefficients) {
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
                                                    targetDatatype,
                                                    std::vector<float>(),
                                                    cldnn::reorder_mean_mode::subtract);

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
                                      op->get_autob());

    p.add_primitive(*op, eltwisePrim);
}

static void CreateAddOp(Program& p, const std::shared_ptr<ngraph::op::v1::Add>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sum);
}

static void CreateMultiplyOp(Program& p, const std::shared_ptr<ngraph::op::v1::Multiply>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::prod);
}

static void CreateMaximumOp(Program& p, const std::shared_ptr<ngraph::op::v1::Maximum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::max);
}

static void CreateMinimumOp(Program& p, const std::shared_ptr<ngraph::op::v1::Minimum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::min);
}

static void CreateSubtractOp(Program& p, const std::shared_ptr<ngraph::op::v1::Subtract>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sub);
}

static void CreateDivideOp(Program& p, const std::shared_ptr<ngraph::op::v1::Divide>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::div);
}

static void CreateSquaredDifferenceOp(Program& p, const std::shared_ptr<ngraph::op::v0::SquaredDifference>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::squared_diff);
}

static void CreateEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::Equal>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::eq);
}

static void CreateNotEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::NotEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ne);
}

static void CreateLessOp(Program& p, const std::shared_ptr<ngraph::op::v1::Less>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::lt);
}

static void CreateLessEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::LessEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::le);
}

static void CreateGreaterOp(Program& p, const std::shared_ptr<ngraph::op::v1::Greater>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::gt);
}

static void CreateGreaterEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::GreaterEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ge);
}

static void CreateLogicalAndOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalAnd>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_and);
}

static void CreateLogicalOrOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalOr>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_or);
}

static void CreateLogicalXorOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalXor>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_xor);
}

static void CreatePowerOp(Program& p, const std::shared_ptr<ngraph::op::v1::Power>& op) {
    validate_inputs_count(op, {2});
    auto power_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (power_node) {
        if (ngraph::shape_size(power_node->get_output_shape(0)) == 1) {
            float pow;
            if (!ov::op::util::get_single_value(power_node, pow))
                IE_THROW() << "Invalid parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
            CreateUnaryEltwiseOp(p, op, cldnn::activation_func::pow, {pow});
            return;
        }
    }
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::pow);
}

static void CreateFloorModOp(Program& p, const std::shared_ptr<ngraph::op::v1::FloorMod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::floor_mod);
}

static void CreateModOp(Program& p, const std::shared_ptr<ngraph::op::v1::Mod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::mod);
}

static void CreateIsFiniteOp(Program& p, const std::shared_ptr<ngraph::op::v10::IsFinite>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_finite);
}

static void CreateIsInfOp(Program& p, const std::shared_ptr<ngraph::op::v10::IsInf>& op) {
    const auto& attributes = op->get_attributes();
    const auto detect_negative = static_cast<float>(attributes.detect_negative);
    const auto detect_positive = static_cast<float>(attributes.detect_positive);
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_inf, {detect_negative, detect_positive});
}

static void CreateIsNaNOp(Program& p, const std::shared_ptr<ngraph::op::v10::IsNaN>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::is_nan);
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

}  // namespace intel_gpu
}  // namespace ov
