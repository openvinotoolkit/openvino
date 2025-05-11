// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"

namespace transpose_sinking {
namespace testing {
namespace utils {

using NodePtr = std::shared_ptr<ov::Node>;

std::string to_string(const ov::Shape& shape);
ov::ParameterVector filter_parameters(const ov::OutputVector& out_vec);

ov::OutputVector set_transpose_for(const std::vector<size_t>& idxs, const ov::OutputVector& out_vec);
ov::OutputVector set_transpose_with_order(const std::vector<size_t>& idxs,
                                          const ov::OutputVector& out_vec,
                                          const std::vector<size_t>& transpose_order_axes);
ov::OutputVector set_gather_for(const std::vector<size_t>& idxs, const ov::OutputVector& out_vec);
std::shared_ptr<ov::Node> create_main_node(const ov::OutputVector& inputs, size_t num_ops, const FactoryPtr& creator);

ov::Output<ov::Node> parameter(ov::element::Type el_type, const ov::PartialShape& ps);
template <class T>
ov::Output<ov::Node> constant(ov::element::Type el_type, const ov::Shape& shape, const std::vector<T>& value) {
    return ov::opset10::Constant::create<T>(el_type, shape, value);
}

}  // namespace utils
}  // namespace testing
}  // namespace transpose_sinking
