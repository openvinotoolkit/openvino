// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "openvino/op/constant.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace utils {
namespace builder {

template<typename T = float>
std::shared_ptr<Node> makeConstant(const ov::element::Type& prc,
                                   const ov::Shape &shape,
                                   const std::vector<T> &data,
                                   bool random = false,
                                   unsigned range = 10, int start_from = 0,
                                   int k = 1, int seed = 1) {
    std::shared_ptr<ov::op::v0::Constant> node;

    if (data.size()) {
        node = std::make_shared<ov::op::v0::Constant>(prc, shape, data);
    } else {
        ov::Tensor data_tensor(prc, shape);
        fill_tensor_random(data_tensor, range, start_from, k, seed);
        node = std::make_shared<ov::op::v0::Constant>(data_tensor);
    }

    return node;
}

}  // namespace builder
}  // namespace utils
}  // namespace test
}  // namespace ov