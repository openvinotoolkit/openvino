// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
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

#include "api/activation.hpp"
#include "api/eltwise.hpp"
#include "api/reorder.hpp"
#include "api/reshape.hpp"

namespace CLDNNPlugin {

void CreateElementwiseOp(Program& p, const std::shared_ptr<ngraph::Node>& op, cldnn::eltwise_mode mode) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outRank = op->get_output_shape(0).size();
    for (size_t i = 0; i < inputPrimitives.size(); ++i) {
        auto inputShape = op->get_input_shape(i);
        auto inputRank = inputShape.size();
        if (inputRank != outRank) {
            // Add reorder if changing number of dimensions requires changing format
            auto targetFormat = DefaultFormatForDims(outRank);
            if (targetFormat.value != DefaultFormatForDims(inputRank).value) {
                auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                auto targetDatatype = DataTypeFromPrecision(op->get_input_element_type(i));
                auto reorderPrim = cldnn::reorder(reorderName, inputPrimitives[i], targetFormat, targetDatatype);

                p.AddPrimitive(reorderPrim);
                p.AddInnerPrimitiveToProfiler(reorderName, layerName, op);

                inputPrimitives[i] = reorderName;
            }

            auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

            // Extend input dimensions by prepending ones
            inputShape.insert(inputShape.begin(), outRank - inputRank, 1ul);

            auto targetShape = CldnnTensorFromIEDims(inputShape);

            auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], targetShape);
            p.AddPrimitive(reshapePrim);
            p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

            inputPrimitives[i] = reshapeName;
        }
    }

    auto out_dt = DataTypeFromPrecision(op->get_output_element_type(0));
    auto eltwisePrim = cldnn::eltwise(layerName,
                                      inputPrimitives,
                                      mode,
                                      {},
                                      out_dt);

    p.AddPrimitive(eltwisePrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateAddOp(Program& p, const std::shared_ptr<ngraph::op::v1::Add>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sum);
}

void CreateMultiplyOp(Program& p, const std::shared_ptr<ngraph::op::v1::Multiply>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::prod);
}

void CreateMaximumOp(Program& p, const std::shared_ptr<ngraph::op::v1::Maximum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::max);
}

void CreateMinimumOp(Program& p, const std::shared_ptr<ngraph::op::v1::Minimum>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::min);
}

void CreateSubtractOp(Program& p, const std::shared_ptr<ngraph::op::v1::Subtract>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::sub);
}

void CreateDivideOp(Program& p, const std::shared_ptr<ngraph::op::v1::Divide>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::div);
}

void CreateSquaredDifferenceOp(Program& p, const std::shared_ptr<ngraph::op::v0::SquaredDifference>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::squared_diff);
}

void CreateEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::Equal>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::eq);
}

void CreateNotEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::NotEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ne);
}

void CreateLessOp(Program& p, const std::shared_ptr<ngraph::op::v1::Less>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::lt);
}

void CreateLessEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::LessEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::le);
}

void CreateGreaterOp(Program& p, const std::shared_ptr<ngraph::op::v1::Greater>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::gt);
}

void CreateGreaterEqualOp(Program& p, const std::shared_ptr<ngraph::op::v1::GreaterEqual>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::ge);
}

void CreateLogicalAndOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalAnd>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_and);
}

void CreateLogicalOrOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalOr>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_or);
}

void CreateLogicalXorOp(Program& p, const std::shared_ptr<ngraph::op::v1::LogicalXor>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::logic_xor);
}

void CreatePowerOp(Program& p, const std::shared_ptr<ngraph::op::v1::Power>& op) {
    p.ValidateInputs(op, {2});
    auto power_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (power_node) {
        if (ngraph::shape_size(power_node->get_output_shape(0)) == 1) {
            float pow;
            if (!ngraph::op::util::get_single_value(power_node, pow))
                THROW_IE_EXCEPTION << "Invalid parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
            CreateUnaryEltwiseOp(p, op, cldnn::activation_func::pow, {pow});
            return;
        }
    }
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::pow);
}

void CreateFloorModOp(Program& p, const std::shared_ptr<ngraph::op::v1::FloorMod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::floor_mod);
}

void CreateModOp(Program& p, const std::shared_ptr<ngraph::op::v1::Mod>& op) {
    CreateElementwiseOp(p, op, cldnn::eltwise_mode::mod);
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

}  // namespace CLDNNPlugin
