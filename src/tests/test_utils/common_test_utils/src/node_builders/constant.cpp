// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"

#include "common_test_utils/type_ranges.hpp"

namespace ov {
namespace test {
namespace utils {

namespace {
void apply_type_range_bound(InputGenerateData& in_data, ov::element::Type type) {
    const auto& type_range = rangeByType.get_range(type);
    auto max = std::min(in_data.start_from + in_data.range, type_range.start_from + type_range.range);
    in_data.start_from = std::max(in_data.start_from, type_range.start_from);
    in_data.range = static_cast<uint32_t>(max - in_data.start_from);
}
}  // namespace

std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const ov::Shape& shape,
                                        InputGenerateData in_data) {
    if (type.is_integral() && ov::element::nf4 != type)
        in_data.resolution = 1;
    apply_type_range_bound(in_data, type);

    auto tensor = create_and_fill_tensor(type, shape, in_data);
    return std::make_shared<ov::op::v0::Constant>(tensor);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
