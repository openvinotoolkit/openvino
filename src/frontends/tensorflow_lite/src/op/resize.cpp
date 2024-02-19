// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::frontend::tensorflow::op;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector resize_bilinear(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    const std::map<std::string, ov::Any> attrs{
        {"align_corners", decoder->get_attribute(&tflite::ResizeBilinearOptions::align_corners)},
        {"half_pixel_centers", decoder->get_attribute(&tflite::ResizeBilinearOptions::half_pixel_centers)},
    };
    return attribute_helper(node, attrs, translate_interpolate_op, "ResizeBilinear");
}

OutputVector resize_nearest_neightbor(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    const std::map<std::string, ov::Any> attrs{
        {"align_corners", decoder->get_attribute(&tflite::ResizeNearestNeighborOptions::align_corners)},
        {"half_pixel_centers", false},
    };
    return attribute_helper(node, attrs, translate_interpolate_op, "ResizeNearestNeighbor");
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
