// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/eye.hpp"

#include <memory>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/eye.hpp"
#include "intel_gpu/runtime/layout.hpp"

namespace ov {
namespace intel_gpu {

namespace {

static void CreateEyeOp(Program& p, const std::shared_ptr<ngraph::op::v9::Eye>& op) {
    validate_inputs_count(op, {3, 4});

    const InferenceEngine::SizeVector& output_shapes = op->get_output_shape(0);
    auto os_sz = output_shapes.size();
    assert(2 <= os_sz && os_sz <= 5);

    size_t dim_size = std::max(os_sz, static_cast<size_t>(4));
    InferenceEngine::SizeVector dims(dim_size, 1);
    for (size_t i = dim_size, j = os_sz; i > 0 && j > 0; --i, --j) {
        dims[i - 1] = output_shapes[j - 1];
    }
    const ngraph::op::v0::Constant* constant = dynamic_cast<const ngraph::op::v0::Constant*>(op->get_input_node_ptr(2));
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
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    auto output_shape = tensor_from_dims(dims);
    auto eye_prim = cldnn::eye(layer_type_name_ID(op),
                               input_primitives,
                               output_shape,
                               shift,
                               cldnn::element_type_to_data_type(op->get_out_type()));

    p.add_primitive(*op, eye_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v9, Eye);

}  // namespace intel_gpu
}  // namespace ov
