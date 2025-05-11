// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/gather_nd.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_gather_nd(const ov::Output<Node>& data_node,
                                         const ov::Shape& indices_shape,
                                         const ov::element::Type& indices_type,
                                         const std::size_t batch_dims) {
    const auto indices = [&] {
        const auto& data_shape = data_node.get_shape();
        const auto indices_count =
            std::accumulate(begin(indices_shape), prev(end(indices_shape)), 1ull, std::multiplies<std::size_t>{});

        const auto slice_rank = indices_shape.back();

        const auto max_dim = *std::max_element(begin(data_shape), end(data_shape));

        std::vector<int> indices_values(indices_count * slice_rank);
        ov::test::utils::fill_data_random(indices_values.data(), indices_count * slice_rank, max_dim, 0);
        auto indices_data = indices_values.data();
        for (int i = 0; i < indices_count; i++) {
            for (int dim = 0; dim < slice_rank; dim++) {
                indices_data[0] = indices_data[0] % data_shape[dim + batch_dims];
                indices_data++;
            }
        }
        return op::v0::Constant::create(indices_type, indices_shape, indices_values);
    }();

    auto gather_nd_node = std::make_shared<ov::op::v5::GatherND>(data_node, indices, batch_dims);
    gather_nd_node->set_friendly_name("GatherND");

    return gather_nd_node;
}

std::shared_ptr<ov::Node> make_gather_nd8(const ov::Output<Node>& data_node,
                                          const ov::Shape& indices_shape,
                                          const ov::element::Type& indices_type,
                                          const std::size_t batch_dims) {
    const auto indices = [&] {
        const auto& data_shape = data_node.get_shape();
        const auto indices_count =
            std::accumulate(begin(indices_shape), prev(end(indices_shape)), 1ull, std::multiplies<std::size_t>{});
        const auto slice_rank = indices_shape.back();

        const auto max_dim = *std::max_element(begin(data_shape), end(data_shape));

        std::vector<int> indices_values(indices_count * slice_rank);
        ov::test::utils::fill_data_random(indices_values.data(), indices_count * slice_rank, max_dim, 0);
        auto indices_data = indices_values.data();
        for (int i = 0; i < indices_count; i++) {
            for (int dim = 0; dim < slice_rank; dim++) {
                indices_data[0] = indices_data[0] % data_shape[dim + batch_dims];
                indices_data++;
            }
        }
        return op::v0::Constant::create(indices_type, indices_shape, indices_values);
    }();

    auto gather_nd_node = std::make_shared<ov::op::v8::GatherND>(data_node, indices, batch_dims);
    gather_nd_node->set_friendly_name("GatherND");

    return gather_nd_node;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
