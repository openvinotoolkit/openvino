// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/primitives/matrix_nms.hpp"

#include "openvino/op/matrix_nms.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"

#include <memory>

namespace ov {
namespace op {
namespace internal {
using NmsStaticShapeIE8 = ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>;
}
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

namespace {
void CreateNmsStaticShapeIE8Op(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::NmsStaticShapeIE8>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);

    auto prim = cldnn::matrix_nms(layer_type_name_ID(op), inputs[0], inputs[1], op->get_attrs());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

    p.add_primitive(*op, prim);
}

}  // anonymous namespace

REGISTER_FACTORY_IMPL(internal, NmsStaticShapeIE8);

}  // namespace intel_gpu
}  // namespace ov
