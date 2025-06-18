// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace ov::op;
using namespace std;

OutputVector translate_erfc_util(const NodeContext& context, const Output<Node>& data) {
    auto one_const = create_same_type_const_scalar<int32_t>(data, 1);
    auto erf = context.mark_node(make_shared<v0::Erf>(data));
    auto erfc = context.mark_node(make_shared<v1::Subtract>(one_const, erf));
    return {erfc};
}

OutputVector translate_erfc(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto data = context.get_input(0);

    return translate_erfc_util(context, data);
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
