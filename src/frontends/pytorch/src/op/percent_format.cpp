// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_percent_format(const NodeContext& context) {

    if (context.get_input_size() < 2) {
        return {context.get_input(0)}; 
    }
    return {context.get_input(1)};
};

}  
} 
}  
}  
