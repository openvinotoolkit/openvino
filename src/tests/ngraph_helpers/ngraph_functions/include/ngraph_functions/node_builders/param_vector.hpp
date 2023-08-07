// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "openvino/op/parameter.hpp"

namespace ov {
namespace test {
namespace utils {
namespace builder {

ov::ParameterVector makeParams(const ov::element::Type &type, const std::vector<std::vector<size_t>> &shapes);

ov::ParameterVector makeDynamicParams(const ov::element::Type &type, const std::vector<ov::PartialShape> &shapes);

ov::ParameterVector makeDynamicParams(const std::vector<ov::element::Type>& types, const std::vector<ov::PartialShape>& shapes);

ov::ParameterVector makeParams(const ov::element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs);

}  // namespace builder
}  // namespace utils
}  // namespace test
}  // namespace ov

// WA for openvino_contrib repo
namespace ngraph {
namespace builder {
using ov::test::utils::builder::makeParams;
using ov::test::utils::builder::makeDynamicParams;
}  // namespace builder
}  // namespace ngraph
