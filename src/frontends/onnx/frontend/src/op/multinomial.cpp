// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/multinomial.hpp"

#include "exceptions.hpp"
#include "openvino/op/multinomial.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;
namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector multinomial(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);

    const auto sample_size = node.get_attribute_as_constant<int64_t>("sample_size", 1);

    const auto dtype =
        node.get_attribute_value<int64_t>("dtype",
                                          static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_INT32));
    const auto seed = common::convert_float_seed(node.get_attribute_value<float>("seed", 0.0f));
    const auto target_type = common::get_ov_element_type(dtype);
    const uint64_t global_seed = 0;

    auto multinomial_op =
        std::make_shared<ov::op::v13::Multinomial>(input, sample_size, target_type, true, true, seed, global_seed);

    return {multinomial_op};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
