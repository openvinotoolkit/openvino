// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_poisson(const NodeContext& context) {
    // aten::poisson(Tensor self, Generator? generator=None) -> Tensor
    num_inputs_check(context, 1, 2);
    
    auto rates = context.get_input(0);
    
    // Check if generator (seed) is provided
    if (!context.input_is_none(1)) {
        PYTORCH_OP_CONVERSION_CHECK(false, 
            "aten::poisson conversion with generator is not supported");
    }
    
    // Use PtFrameworkNode as a passthrough
    auto fw_node = std::make_shared<PtFrameworkNode>(
        context.get_decoder(), 
        OutputVector{rates}, 
        1
    );
    
    // Set output type same as input
    fw_node->set_output_type(0, 
        rates.get_element_type(), 
        rates.get_partial_shape()
    );
    
    auto res = context.mark_node(fw_node);
    return {res};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
```

4. **Save it**

---

## Step 4: Register the Operation

Now modify `op_table.cpp`:

1. Open:
```
   C:\Users\ACER\Desktop\Openvino\openvino\src\frontends\pytorch\src\op_table.cpp