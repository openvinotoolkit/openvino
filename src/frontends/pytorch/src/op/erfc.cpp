// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_erfc(const NodeContext& context) {
    // aten::erf(Tensor self) -> Tensor
    // aten::erf.out(Tensor self, Tensor(!a) out) -> Tensor(!a)
    num_inputs_check(context, 1, 2);
    auto x = get_input_with_floating_type(context, 0);

    auto y = common_translators::translate_erfc_util(context, x)[0];

    if (!context.input_is_none(1)) {
        context.mutate_input(1, y);
    }
    return {y};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
