// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const ov::Shape& shape,
                                        const InputGenerateData& in_data) {
    auto tensor = create_and_fill_tensor(type, shape, in_data);
    return std::make_shared<ov::op::v0::Constant>(tensor);
}

std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type, const ov::Shape& shape) {
    InputGenerateData in_data(1, 9);
    if (type.is_real())
        in_data.resolution = 1000;
    return make_constant(type, shape, in_data);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
