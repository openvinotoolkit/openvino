// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/fake_convert.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/fake_convert.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fake_convert(const ov::Output<ov::Node>& in,
                                            const ov::Output<ov::Node>& scale,
                                            const ov::Output<ov::Node>& shift,
                                            ov::element::Type destination_type) {
    auto fc = std::make_shared<ov::op::v13::FakeConvert>(in, scale, shift, destination_type);
    return fc;
}

std::shared_ptr<ov::Node> make_fake_convert(const ov::Output<ov::Node>& in,
                                            const ov::Output<ov::Node>& scale,
                                            ov::element::Type destination_type) {
    auto fc = std::make_shared<ov::op::v13::FakeConvert>(in, scale, destination_type);
    return fc;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
