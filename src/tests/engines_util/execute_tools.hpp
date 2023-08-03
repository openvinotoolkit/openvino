// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>

#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/op.hpp"

bool validate_list(const std::vector<std::shared_ptr<ov::Node>>& nodes);
std::shared_ptr<ov::Model> make_test_graph();

OPENVINO_SUPPRESS_DEPRECATED_START
template <typename T>
void copy_data(const std::shared_ptr<ngraph::runtime::Tensor>& tv, const std::vector<T>& data) {
    size_t data_size = data.size() * sizeof(T);
    if (data_size > 0) {
        tv->write(data.data(), data_size);
    }
}

template <>
inline void copy_data<bool>(const std::shared_ptr<ngraph::runtime::Tensor>& tv, const std::vector<bool>& data) {
    std::vector<char> data_char(data.begin(), data.end());
    copy_data(tv, data_char);
}

template <ov::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ov::Shape& shape,
                                       const std::vector<typename ov::element_type_traits<ET>::value_type>& data) {
    NGRAPH_CHECK(shape_size(shape) == data.size(), "Incorrect number of initialization elements");
    auto host_tensor = std::make_shared<ngraph::runtime::HostTensor>(ET, shape);
    copy_data(host_tensor, data);
    return host_tensor;
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine);

template <ov::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ov::Shape& shape) {
    auto host_tensor = std::make_shared<ngraph::runtime::HostTensor>(ET, shape);
    static std::default_random_engine engine(2112);
    random_init(host_tensor.get(), engine);
    return host_tensor;
}
OPENVINO_SUPPRESS_DEPRECATED_END

testing::AssertionResult test_ordered_ops(std::shared_ptr<ov::Model> f, const ov::NodeVector& required_ops);
