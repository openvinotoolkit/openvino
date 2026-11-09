// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov::frontend::tensorflow_lite::op {

OutputVector unique(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return indexed_from_named(ov::frontend::tensorflow::op::translate_unique_op(node));
}

}  // namespace ov::frontend::tensorflow_lite::op
