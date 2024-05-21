// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_ops/rfft2d.h"

#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector rfft2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = node.get_decoder();
    auto rfft = make_shared<ov::frontend::tensorflow_lite::Rfft2d>(node.get_input(0), node.get_input(1), decoder);
    rfft->set_friendly_name(decoder->get_op_name());
    return rfft->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
