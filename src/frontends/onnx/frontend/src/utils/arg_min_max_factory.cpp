// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/arg_min_max_factory.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"

using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace utils {

ArgMinMaxFactory::ArgMinMaxFactory(const Node& node)
    : m_keep_dims{node.get_attribute_value<std::int64_t>("keepdims", 1)},
      m_input_node{node.get_ov_inputs().at(0)},
      m_axis{node.get_attribute_value<std::int64_t>("axis", 0)},
      m_select_last_index{node.get_attribute_value<std::int64_t>("select_last_index", 0)} {}

std::shared_ptr<ov::Node> ArgMinMaxFactory::make_arg_max() const {
    return make_topk_subgraph(v11::TopK::Mode::MAX);
}

std::shared_ptr<ov::Node> ArgMinMaxFactory::make_arg_min() const {
    return make_topk_subgraph(v11::TopK::Mode::MIN);
}

std::shared_ptr<ov::Node> ArgMinMaxFactory::make_topk_subgraph(v11::TopK::Mode mode) const {
    const auto k_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    if (m_select_last_index == 1) {
        // Example (ArgMin):
        // The goal is to get the index of the last occurrence of the
        // minimum value present in given input tensor.
        //
        // Input:           [1, 2, 1, 3, 4, 4]
        // Expected output: [2]
        //
        // Top-K is always returning the "most-left" result. The trick is to
        // reverse input to find the "most-right" occurrence which is equal to
        // the last occurrence in the original input.
        // reverse = [4, 4, 3, 1, 2, 1]
        //
        // Run TopK on reversed tensor, in the example output with index values
        // is equal to:
        // topk->output(1) = 3
        //
        // Using ShapeOf and Gather on input obtain length of the input tensor
        // along axis, in the example this is equal to:
        // dims_on_axis = 6
        //
        // Now using two Substract ops calculate resulting index:
        // res_index = dims_on_axis - topk->output(1) = 6 - 3 = 3
        // result = res_index - 1 = 3 - 1 = 2

        const int64_t normalized_axis =
            ov::util::try_normalize_axis(m_axis, m_input_node.get_partial_shape().rank(), *m_input_node.get_node());

        const auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {normalized_axis});
        const auto reverse = std::make_shared<v1::Reverse>(m_input_node, axis_node, v1::Reverse::Mode::INDEX);

        const auto topk = std::make_shared<v11::TopK>(reverse, k_node, normalized_axis, mode, v1::TopK::SortType::NONE);

        const auto data_shape = std::make_shared<v0::ShapeOf>(m_input_node);
        const auto dims_on_axis =
            std::make_shared<v1::Gather>(data_shape,
                                         axis_node,
                                         v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));

        const auto res_index =
            std::make_shared<v1::Subtract>(dims_on_axis,
                                           std::make_shared<v0::Convert>(topk->output(1), ov::element::i64));
        const auto result =
            std::make_shared<v1::Subtract>(res_index, v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        if (m_keep_dims == 0) {
            const auto axis_to_remove = v0::Constant::create(ov::element::u64, ov::Shape{}, {topk->get_axis()});

            return std::make_shared<v0::Squeeze>(result, axis_to_remove);
        }

        return result;
    }

    const auto topk = std::make_shared<v11::TopK>(m_input_node, k_node, m_axis, mode, v11::TopK::SortType::NONE);

    const auto result = std::make_shared<v0::Convert>(topk->output(1), ov::element::i64);

    if (m_keep_dims == 0) {
        const auto axis_to_remove = v0::Constant::create(ov::element::u64, ov::Shape{}, {topk->get_axis()});

        return std::make_shared<v0::Squeeze>(result, axis_to_remove);
    }

    return result;
}
}  // namespace utils
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
