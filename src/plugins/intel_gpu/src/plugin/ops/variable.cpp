// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/read_value.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"


namespace ov {
namespace intel_gpu {

namespace {
template<typename T_PRIMITIVE>
void CreateVariableAccessPrimitive(ProgramBuilder &p, const std::shared_ptr<ov::op::Op> &op,
                                   const std::string &variable_id) {
    validate_inputs_count(op, {1});

    const auto output_pshape = op->get_output_partial_shape(0);
    const auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    const auto output_format = cldnn::format::get_default_format(output_pshape.size());

    const auto variable_layout = cldnn::layout{ output_pshape, output_dtype, output_format };

    auto inputs = p.GetInputInfo(op);
    if (!p.use_new_shape_infer())
        p.AddVariableStateInfo(variable_id, variable_layout);
    const auto prim = T_PRIMITIVE{layer_type_name_ID(op),
                                  inputs,
                                  variable_id,
                                  variable_layout};

    p.add_primitive(*op, prim);
}

void CreateReadValueOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ReadValue>& op) {
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateAssignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Assign>& op) {
    CreateVariableAccessPrimitive<cldnn::assign>(p, op, op->get_variable_id());
}

void CreateReadValueOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::ReadValue>& op) {
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateAssignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::Assign>& op) {
    CreateVariableAccessPrimitive<cldnn::assign>(p, op, op->get_variable_id());
}

} // namespace

REGISTER_FACTORY_IMPL(v3, Assign);
REGISTER_FACTORY_IMPL(v6, Assign);
REGISTER_FACTORY_IMPL(v3, ReadValue);
REGISTER_FACTORY_IMPL(v6, ReadValue);

}  // namespace intel_gpu
}  // namespace ov
