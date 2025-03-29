// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_no_op(const NodeContext& node) {
    // the operation does nothing in terms of data generation
    default_op_checks(node, 0, {"NoOp", "SaveV2", "Assert", "LookupTableInsert", "LookupTableInsertV2"});
    return {};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
