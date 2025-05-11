// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/dft.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/idft.hpp"

namespace ov {
namespace test {
namespace utils {
namespace {
template <typename... Args>
std::shared_ptr<ov::Node> CallDftCtorWithArgs(const ov::test::utils::DFTOpType op_type, Args&&... args) {
    switch (op_type) {
    case ov::test::utils::DFTOpType::FORWARD:
        return std::make_shared<ov::op::v7::DFT>(std::forward<Args>(args)...);
    case ov::test::utils::DFTOpType::INVERSE:
        return std::make_shared<ov::op::v7::IDFT>(std::forward<Args>(args)...);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}
}  // namespace

std::shared_ptr<ov::Node> make_dft(ov::ParameterVector& parameters,
                                   const std::vector<int64_t>& axes,
                                   const std::vector<int64_t>& signal_size,
                                   const ov::test::utils::DFTOpType op_type,
                                   utils::InputLayerType axes_in_type,
                                   utils::InputLayerType size_in_type) {
    std::shared_ptr<ov::Node> signal_size_node;
    std::shared_ptr<ov::Node> axes_node;

    if (axes_in_type == utils::InputLayerType::PARAMETER) {
        axes_node = parameters.emplace_back(std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{axes.size()}));
    } else {
        axes_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{axes.size()}, axes);
    }

    if (!signal_size.empty()) {
        if (size_in_type == utils::InputLayerType::PARAMETER) {
            signal_size_node = parameters.emplace_back(
                std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{signal_size.size()}));
        } else {
            signal_size_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                      ov::Shape{signal_size.size()},
                                                                      signal_size);
        }
        return CallDftCtorWithArgs(op_type, parameters[0], axes_node, signal_size_node);
    }
    return CallDftCtorWithArgs(op_type, parameters[0], axes_node);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
