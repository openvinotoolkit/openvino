// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lite_op_table.hpp"
#include "op_table.hpp"

#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {


std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
    };
};

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov