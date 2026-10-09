// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

template <typename T>
std::vector<ov::Tensor> calculate_selective_ssm_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                                                     const std::shared_ptr<ov::Model>& function) {
    const auto& params = function->get_parameters();
    const auto& A_tensor = inputs.at(params[0]);
    const auto& dt_tensor = inputs.at(params[1]);
    const auto& B_tensor = inputs.at(params[2]);
    const auto& x_tensor = inputs.at(params[3]);
    const auto& C_tensor = inputs.at(params[4]);
    const auto& initial_state_tensor = inputs.at(params[5]);
    const auto& x_shape = x_tensor.get_shape();
    const auto& B_shape = B_tensor.get_shape();
    const auto& state_shape = initial_state_tensor.get_shape();
    const auto batch_size = x_shape[0];
    const auto sequence_length = x_shape[1];
    const auto num_heads = x_shape[2];
    const auto head_dim = x_shape[3];
    const auto num_groups = B_shape[2];
    const auto state_size = B_shape[3];
    const auto heads_per_group = num_heads / num_groups;

    const auto* A = A_tensor.data<const T>();
    const auto* dt = dt_tensor.data<const T>();
    const auto* B = B_tensor.data<const T>();
    const auto* x = x_tensor.data<const T>();
    const auto* C = C_tensor.data<const T>();
    const auto* initial_state = initial_state_tensor.data<const T>();
    std::vector<float> state(initial_state_tensor.get_size());
    std::transform(initial_state, initial_state + state.size(), state.begin(), [](T value) {
        return static_cast<float>(value);
    });

    // Deliberately materialize the small discretized terms in the oracle. The production kernel does not use this
    // decomposition, so this catches indexing/grouping mistakes independently of its fused update-reduction loop.
    const auto token_heads = batch_size * sequence_length * num_heads;
    std::vector<float> decay(token_heads);
    std::vector<float> discretized_B(token_heads * state_size);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t token = 0; token < sequence_length; ++token) {
            for (size_t head = 0; head < num_heads; ++head) {
                const auto token_head = (batch * sequence_length + token) * num_heads + head;
                const auto group = head / heads_per_group;
                const auto grouped_projection =
                    ((batch * sequence_length + token) * num_groups + group) * state_size;
                const float delta = static_cast<float>(dt[token_head]);
                decay[token_head] = std::exp(static_cast<float>(A[head]) * delta);
                for (size_t state_index = 0; state_index < state_size; ++state_index) {
                    discretized_B[token_head * state_size + state_index] =
                        delta * static_cast<float>(B[grouped_projection + state_index]);
                }
            }
        }
    }

    ov::Tensor output_tensor(x_tensor.get_element_type(), x_shape);
    auto* output = output_tensor.data<T>();
    const auto state_batch_stride = num_heads * head_dim * state_size;
    const auto state_head_stride = head_dim * state_size;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t head = 0; head < num_heads; ++head) {
            const auto group = head / heads_per_group;
            for (size_t position = 0; position < head_dim; ++position) {
                const auto state_base =
                    batch * state_batch_stride + head * state_head_stride + position * state_size;
                for (size_t token = 0; token < sequence_length; ++token) {
                    const auto token_head = (batch * sequence_length + token) * num_heads + head;
                    const auto projection_base =
                        ((batch * sequence_length + token) * num_groups + group) * state_size;
                    const auto x_index = token_head * head_dim + position;
                    const float input = static_cast<float>(x[x_index]);
                    for (size_t state_index = 0; state_index < state_size; ++state_index) {
                        auto& state_value = state[state_base + state_index];
                        state_value = state_value * decay[token_head] +
                                      input * discretized_B[token_head * state_size + state_index];
                    }

                    double reduction = 0.0;
                    for (size_t state_index = 0; state_index < state_size; ++state_index) {
                        reduction += static_cast<double>(state[state_base + state_index]) *
                                     static_cast<float>(C[projection_base + state_index]);
                    }
                    output[x_index] = static_cast<T>(reduction);
                }
            }
        }
    }

    ov::Tensor state_tensor(initial_state_tensor.get_element_type(), state_shape);
    std::transform(state.begin(), state.end(), state_tensor.data<T>(), [](float value) {
        return static_cast<T>(value);
    });
    return {output_tensor, state_tensor};
}
}  // namespace

namespace ov::test {

std::string SelectiveSSM::getTestCaseName(const testing::TestParamInfo<selective_ssm_params>& obj) {
    const auto& [batch, seq_len, num_heads, num_groups, head_dim, state_size, prec, device] = obj.param;
    std::ostringstream result;
    result << "batch=" << batch;
    result << ",seq_len=" << seq_len;
    result << ",num_heads=" << num_heads;
    result << ",num_groups=" << num_groups;
    result << ",head_dim=" << head_dim;
    result << ",state_size=" << state_size;
    result << ",prec=" << prec;
    result << ",device=" << device;
    return result.str();
}

void SelectiveSSM::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto& shape = targetInputStaticShapes[i];
        if (i == 0) {
            inputs[param] = ov::test::utils::create_and_fill_tensor_real_distribution(param->get_element_type(),
                                                                                      shape,
                                                                                      -0.5f,
                                                                                      0.2f,
                                                                                      1);
        } else if (i == 1) {
            inputs[param] = ov::test::utils::create_and_fill_tensor_real_distribution(param->get_element_type(),
                                                                                      shape,
                                                                                      0.0f,
                                                                                      0.5f,
                                                                                      1);
        } else {
            inputs[param] =
                ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                        shape,
                                                        ov::test::utils::InputGenerateData(-0.5f, 1.0f, 1000, 1));
        }
    }
}

std::vector<ov::Tensor> SelectiveSSM::calculate_refs() {
    const auto& precision = std::get<6>(GetParam());
    if (precision == ov::element::f16) {
        return calculate_selective_ssm_refs<ov::float16>(inputs, function);
    }
    if (precision == ov::element::bf16) {
        return calculate_selective_ssm_refs<ov::bfloat16>(inputs, function);
    }
    return calculate_selective_ssm_refs<float>(inputs, function);
}

void SelectiveSSM::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
}

void SelectiveSSM::SetUp() {
    const auto& [batch, seq_len, num_heads, num_groups, head_dim, state_size, prec, device] = GetParam();

    targetDevice = device;
    inType = prec;
    configuration[ov::hint::inference_precision.name()] = prec;

    abs_threshold = prec == ov::element::f32 ? 1e-6f : 1e-3f;
    rel_threshold = 1e-5f;

    const ov::Shape A_shape{static_cast<size_t>(num_heads)};
    const ov::Shape dt_shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_heads)};
    const ov::Shape alternate_dt_shape{static_cast<size_t>(batch == 1 ? 2 : 1),
                                       static_cast<size_t>(seq_len + 2),
                                       static_cast<size_t>(num_heads)};
    const ov::Shape B_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(num_groups),
                            static_cast<size_t>(state_size)};
    const ov::Shape x_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(num_heads),
                            static_cast<size_t>(head_dim)};
    const ov::Shape alternate_B_shape{alternate_dt_shape[0],
                                      alternate_dt_shape[1],
                                      static_cast<size_t>(num_groups),
                                      static_cast<size_t>(state_size)};
    const ov::Shape alternate_x_shape{alternate_dt_shape[0],
                                      alternate_dt_shape[1],
                                      static_cast<size_t>(num_heads),
                                      static_cast<size_t>(head_dim)};
    const ov::Shape state_shape{static_cast<size_t>(batch),
                                static_cast<size_t>(num_heads),
                                static_cast<size_t>(head_dim),
                                static_cast<size_t>(state_size)};
    const ov::Shape alternate_state_shape{alternate_dt_shape[0],
                                          static_cast<size_t>(num_heads),
                                          static_cast<size_t>(head_dim),
                                          static_cast<size_t>(state_size)};

    init_input_shapes(
        {InputShape{ov::PartialShape{num_heads}, {A_shape, A_shape, A_shape}},
         InputShape{ov::PartialShape{-1, -1, num_heads}, {dt_shape, alternate_dt_shape, dt_shape}},
         InputShape{ov::PartialShape{-1, -1, num_groups, state_size}, {B_shape, alternate_B_shape, B_shape}},
         InputShape{ov::PartialShape{-1, -1, num_heads, head_dim}, {x_shape, alternate_x_shape, x_shape}},
         InputShape{ov::PartialShape{-1, -1, num_groups, state_size}, {B_shape, alternate_B_shape, B_shape}},
         InputShape{ov::PartialShape{-1, num_heads, head_dim, state_size},
                    {state_shape, alternate_state_shape, state_shape}}});

    auto A = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{num_heads});
    auto dt = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{-1, -1, num_heads});
    auto B = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{-1, -1, num_groups, state_size});
    auto x = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{-1, -1, num_heads, head_dim});
    auto C = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{-1, -1, num_groups, state_size});
    auto state = std::make_shared<ov::op::v0::Parameter>(prec, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto selective_ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    function = std::make_shared<ov::Model>(selective_ssm->outputs(),
                                           ov::ParameterVector{A, dt, B, x, C, state},
                                           "SelectiveSSM");
}

TEST_P(SelectiveSSM, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    const auto runtime_model = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(runtime_model, {"SelectiveSSM"}, 1);
    CheckNumberOfNodesWithType(runtime_model, {"Loop"}, 0);
}

}  // namespace ov::test
