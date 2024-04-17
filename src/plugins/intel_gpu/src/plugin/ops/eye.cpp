// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"
#include "openvino/op/constant.hpp"

#include <memory>

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/eye.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace ov {
namespace intel_gpu {

namespace {

static void CreateEyeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v9::Eye>& op) {
    validate_inputs_count(op, {3, 4});

    const ov::op::v0::Constant* constant = dynamic_cast<ov::op::v0::Constant*>(op->get_input_node_ptr(2));
    OPENVINO_ASSERT(constant != nullptr, "Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    int32_t shift{};
    switch (constant->get_element_type()) {
    case ov::element::Type_t::i32:
        shift = *constant->get_data_ptr<int32_t>();
        break;
    case ov::element::Type_t::i64:
        shift = *constant->get_data_ptr<int64_t>();
        break;
    default:
        throw std::runtime_error{"Input type can be only either i32 or i64"};
        break;
    }
    auto input_info = p.GetInputInfo(op);
    auto eye_prim = cldnn::eye(layer_type_name_ID(op),
                               input_info,
                               shift,
                               cldnn::element_type_to_data_type(op->get_out_type()));

    p.add_primitive(*op, eye_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v9, Eye);

}  // namespace intel_gpu
}  // namespace ov
