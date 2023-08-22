// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/reverse_sequence.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector reverse_sequence(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const auto sequence_lengths = node.get_ng_inputs().at(1);
    // nGraph supports only int32 type of sequence_lengths
    const auto sequence_lengths_i32 =
        std::make_shared<default_opset::Convert>(node.get_ng_inputs().at(1), element::i32);
    const auto data_rank = data.get_partial_shape().rank();

    const auto batch_axis = node.get_attribute_value<int64_t>("batch_axis", 1);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto normalized_batch_axis = ngraph::normalize_axis(node.get_description(), batch_axis, data_rank);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto time_axis = node.get_attribute_value<int64_t>("time_axis", 0);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto normalized_time_axis = ngraph::normalize_axis(node.get_description(), time_axis, data_rank);
    OPENVINO_SUPPRESS_DEPRECATED_END

    NGRAPH_CHECK(normalized_batch_axis == 0 || normalized_batch_axis == 1,
                 "Allowed values of the 'batch_axis' attribute for ReverseSequence "
                 "operator are 0 and 1");

    NGRAPH_CHECK(normalized_time_axis == 0 || normalized_time_axis == 1,
                 "Allowed values of the 'time_axis' attribute for ReverseSequence "
                 "operator are 0 and 1");

    NGRAPH_CHECK(normalized_batch_axis != normalized_time_axis,
                 "'batch_axis' and 'time_axis' attributes of the ReverseSequence "
                 "operator can't point to the same dimension");

    return {std::make_shared<default_opset::ReverseSequence>(data,
                                                             sequence_lengths_i32,
                                                             normalized_batch_axis,
                                                             normalized_time_axis)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
