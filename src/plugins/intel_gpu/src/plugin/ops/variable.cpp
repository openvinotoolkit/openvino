// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "ngraph/op/assign.hpp"
#include "ngraph/op/read_value.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"


namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {
template<typename T_PRIMITIVE>
void CreateVariableAccessPrimitive(Program &p, const std::shared_ptr<ngraph::op::Op> &op,
                                   const std::string &variable_id) {
    p.ValidateInputs(op, {1});

    const auto output_data_type = DataTypeFromPrecision(op->get_output_element_type(0));
    const auto op_output_shape = op->get_output_shape(0);
    const auto output_format = DefaultFormatForDims(op_output_shape.size());
    const auto output_shape = tensor_from_dims(op_output_shape);

    const auto variable_layout = cldnn::layout{output_data_type,
                                               output_format,
                                               output_shape};

    auto input_primitives = p.GetInputPrimitiveIDs(op);
    p.AddVariableStateInfo(variable_id, variable_layout);
    const auto prim = T_PRIMITIVE{layer_type_name_ID(op),
                                  input_primitives,
                                  variable_id,
                                  variable_layout};

    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op);
}

void CreateReadValueOp(Program& p, const std::shared_ptr<ngraph::op::v6::ReadValue>& op) {
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateAssignOp(Program& p, const std::shared_ptr<ngraph::op::v6::Assign>& op) {
    CreateVariableAccessPrimitive<cldnn::assign>(p, op, op->get_variable_id());
}

} // namespace

REGISTER_FACTORY_IMPL(v6, Assign);
REGISTER_FACTORY_IMPL(v6, ReadValue);

} // namespace intel_gpu
} // namespace runtime
} // namespace ov
