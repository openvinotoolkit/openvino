// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph {
class TestOpMultiOut : public op::Op {
public:
    static constexpr NodeTypeInfo type_info{"TestOpMultiOut", static_cast<uint64_t>(0)};
    const NodeTypeInfo& get_type_info() const override {
        return type_info;
    }
    TestOpMultiOut() = default;

    TestOpMultiOut(const Output<Node>& output_1, const Output<Node>& output_2) : Op({output_1, output_2}) {
        validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_size(2);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        set_output_type(1, get_input_element_type(1), get_input_partial_shape(1));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<TestOpMultiOut>(new_args.at(0), new_args.at(1));
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
};
}  // namespace ngraph

bool validate_list(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);
std::shared_ptr<ngraph::Function> make_test_graph();

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& data) {
    size_t data_size = data.size() * sizeof(T);
    if (data_size > 0) {
        tv->write(data.data(), data_size);
    }
}

template <ngraph::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ngraph::Shape& shape,
                                       const std::vector<typename ngraph::element_type_traits<ET>::value_type>& data) {
    NGRAPH_CHECK(shape_size(shape) == data.size(), "Incorrect number of initialization elements");
    auto host_tensor = std::make_shared<ngraph::HostTensor>(ET, shape);
    copy_data(host_tensor, data);
    return host_tensor;
}

template <>
void copy_data<bool>(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<bool>& data);

template <typename T>
void write_vector(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& values) {
    tv->write(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<std::shared_ptr<T>> get_ops_of_type(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<T>> ops;
    for (auto op : f->get_ops()) {
        if (auto cop = ngraph::as_type_ptr<T>(op)) {
            ops.push_back(cop);
        }
    }

    return ops;
}

template <typename T>
void init_int_tv(ngraph::runtime::Tensor* tv, std::default_random_engine& engine, T min, T max) {
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec) {
        element = dist(engine);
    }
    tv->write(vec.data(), vec.size() * sizeof(T));
}

template <typename T>
void init_real_tv(ngraph::runtime::Tensor* tv, std::default_random_engine& engine, T min, T max) {
    size_t size = tv->get_element_count();
    std::uniform_real_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec) {
        element = dist(engine);
    }
    tv->write(vec.data(), vec.size() * sizeof(T));
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine);

template <typename T>
std::string get_results_str(const std::vector<T>& ref_data,
                            const std::vector<T>& actual_data,
                            size_t max_results = 16) {
    std::stringstream ss;
    size_t num_results = std::min(static_cast<size_t>(max_results), ref_data.size());
    ss << "First " << num_results << " results";
    for (size_t i = 0; i < num_results; ++i) {
        ss << std::endl
           // use unary + operator to force integral values to be displayed as numbers
           << std::setw(4) << i << " ref: " << std::setw(16) << std::left << +ref_data[i]
           << "  actual: " << std::setw(16) << std::left << +actual_data[i];
    }
    ss << std::endl;

    return ss.str();
}

template <>
std::string get_results_str(const std::vector<char>& ref_data,
                            const std::vector<char>& actual_data,
                            size_t max_results);

testing::AssertionResult test_ordered_ops(std::shared_ptr<ngraph::Function> f, const ngraph::NodeVector& required_ops);

template <ngraph::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ngraph::Shape& shape) {
    auto host_tensor = std::make_shared<ngraph::HostTensor>(ET, shape);
    static std::default_random_engine engine(2112);
    random_init(host_tensor.get(), engine);
    return host_tensor;
}
