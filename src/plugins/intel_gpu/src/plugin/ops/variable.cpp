// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/read_value.hpp"
#include "transformations/rt_info/original_precision_attribute.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"

namespace ov {
namespace op {
namespace internal {
using ReadValue = ov::intel_gpu::op::ReadValue;
using ReadValues = ov::intel_gpu::op::ReadValues;
}  // namespace internal
}  // namespace op
}  // namespace ov


namespace ov {
namespace intel_gpu {

namespace {
template<typename T_PRIMITIVE>
void CreateVariableAccessPrimitive(ProgramBuilder &p, const std::shared_ptr<ov::op::Op> &op,
                                   const std::string &variable_id) {
    const auto output_pshape = op->get_output_partial_shape(0);
    const auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    const auto output_format = cldnn::format::get_default_format(output_pshape.size());

    const auto variable_layout = cldnn::layout{ output_pshape, output_dtype, output_format };

    auto inputs = p.GetInputInfo(op);
    auto user_specified_type = get_original_precision(op);
    const auto prim = T_PRIMITIVE{layer_type_name_ID(op),
                                  inputs,
                                  variable_id,
                                  { variable_layout },
                                  user_specified_type};

    p.add_primitive(*op, prim);
}

void CreateReadValueOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ReadValue>& op) {
    validate_inputs_count(op, {0, 1});
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateReadValueOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::ReadValue>& op) {
    validate_inputs_count(op, {0, 1});
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateAssignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Assign>& op) {
    validate_inputs_count(op, {1});
    CreateVariableAccessPrimitive<cldnn::assign>(p, op, op->get_variable_id());
}

void CreateReadValueOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::ReadValue>& op) {
    validate_inputs_count(op, {0, 1});
    CreateVariableAccessPrimitive<cldnn::read_value>(p, op, op->get_variable_id());
}

void CreateAssignOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::Assign>& op) {
    validate_inputs_count(op, {1});
    CreateVariableAccessPrimitive<cldnn::assign>(p, op, op->get_variable_id());
}

void CreateReadValuesOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::ReadValues>& op) {
    std::vector<cldnn::layout> variable_layouts;
    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto output_pshape = op->get_output_partial_shape(i);
        const auto output_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(i));
        const auto output_format = cldnn::format::get_default_format(output_pshape.size());
        variable_layouts.emplace_back(output_pshape, output_dtype, output_format);
    }

    auto inputs = p.GetInputInfo(op);
    auto user_specified_type = get_original_precision(op);
    auto prim = cldnn::read_value{layer_type_name_ID(op),
                                  inputs,
                                  op->get_variable_id(),
                                  variable_layouts,
                                  user_specified_type};

    p.add_primitive(*op, prim);
}

} // namespace

REGISTER_FACTORY_IMPL(v3, Assign);
REGISTER_FACTORY_IMPL(v6, Assign);
REGISTER_FACTORY_IMPL(v3, ReadValue);
REGISTER_FACTORY_IMPL(v6, ReadValue);
REGISTER_FACTORY_IMPL(internal, ReadValue);
REGISTER_FACTORY_IMPL(internal, ReadValues);

}  // namespace intel_gpu
}  // namespace ov
