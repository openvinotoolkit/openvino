// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset10.hpp"
#include "transformations/init_node_info.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/gather_sinking.hpp"

using namespace ov;
using namespace ov::opset10;

namespace {
void ShiftLeft(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.begin(), vec.begin() + k, buffer.begin());

    for (int i = k; i < vec.size(); ++i) {
        vec[i - k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.end() - k);
}

void ShiftRight(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.end() - k, vec.end(), buffer.begin());

    for (int i = vec.size() - 1 - k; i >= 0; --i) {
        vec[i + k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.begin());
}

std::vector<size_t> GatherForward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    ShiftLeft(vec, 2);
    return vec;
}

std::vector<size_t> GatherBackward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);  // Not the same as in binary tests
    ShiftRight(vec, 2);
    return vec;
}

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;
using Output = ov::Output<ov::Node>;

template <typename CreateIndicesF>
std::shared_ptr<Gather> MakeGather(NodePtr input_node, CreateIndicesF create_indices_func, size_t axis) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    const std::vector<size_t> indexes = create_indices_func(input_shape[axis], 0);

    auto gather_indexes_node = Constant::create(element::i64, ov::Shape{indexes.size()}, indexes);

    auto gather_axis_node = Constant::create(element::i64, Shape{}, {axis});

    return std::make_shared<Gather>(input_node->output(0), gather_indexes_node, gather_axis_node);
}
}  // namespace

TEST(GatherSinkingGeneral, General) {
    std::shared_ptr<Model> function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather = MakeGather(input_params1, GatherForward, /* axis */ 1);

        auto tanh = std::make_shared<Tanh>(input_params2);
        auto mult = std::make_shared<Multiply>(gather, tanh);
        auto sinh = std::make_shared<Sinh>(mult);

        const auto result = std::make_shared<Result>(sinh);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingGeneral>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather1 = MakeGather(input_params2, GatherBackward, /* axis */ 1);

        auto tanh = std::make_shared<Tanh>(gather1);
        auto mult = std::make_shared<Multiply>(input_params1, tanh);
        auto sinh = std::make_shared<Sinh>(mult);

        auto gather2 = MakeGather(sinh, GatherForward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather2);
        reference_function =
            std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
