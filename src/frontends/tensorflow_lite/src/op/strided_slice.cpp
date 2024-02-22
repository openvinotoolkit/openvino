// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector strided_slice(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    std::map<std::string, ov::Any> attrs{
        {"begin_mask", static_cast<int64_t>(decoder->get_attribute(&tflite::StridedSliceOptions::begin_mask))},
        {"end_mask", static_cast<int64_t>(decoder->get_attribute(&tflite::StridedSliceOptions::end_mask))},
        {"new_axis_mask", static_cast<int64_t>(decoder->get_attribute(&tflite::StridedSliceOptions::new_axis_mask))},
        {"ellipsis_mask", static_cast<int64_t>(decoder->get_attribute(&tflite::StridedSliceOptions::ellipsis_mask))},
        {"shrink_axis_mask",
         static_cast<int64_t>(decoder->get_attribute(&tflite::StridedSliceOptions::shrink_axis_mask))},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_strided_slice_op);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
