// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_ops/complex_abs.h"

#include "op_translation_utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector complex_abs(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto decoder = node.get_decoder();
    auto abs = make_shared<ov::frontend::tensorflow_lite::ComplexAbs>(node.get_input(0), decoder);
    abs->set_friendly_name(decoder->get_op_name());
    return abs->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
