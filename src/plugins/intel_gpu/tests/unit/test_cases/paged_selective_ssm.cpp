// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_selective_ssm.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "paged_selective_ssm_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr int32_t large_state_size = 32 * 1024 + 1;

enum class expected_ssm_impl { any, jit, fallback };

template <typename T>
std::vector<T> make_test_values(size_t count, float scale, float shift = 0.f) {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = static_cast<T>((static_cast<int32_t>(i % 11) - 5) * scale + shift);
    return values;
}

struct paged_selective_ssm_test_params {
    std::vector<int32_t> seq_tokens;
    std::vector<int32_t> processed_tokens;
    std::vector<int32_t> cache_intervals;
    int32_t num_heads;
    int32_t num_groups;
    int32_t head_dim;
    int32_t state_size;
    ov::element::Type precision;
    ov::element::Type index_precision;
    bool dynamic_shapes;
    std::vector<bool> alias_first_write;
    bool reverse_blocks = false;
    bool caching_test = false;
    bool padded_layouts = false;
    int32_t iterations = 1;
    int32_t invalid_metadata = 0;
    bool accumulation_test = false;
    float relative_output_tolerance = 5e-5f;
    ov::element::Type state_precision = ov::element::dynamic;
    expected_ssm_impl expected_impl = expected_ssm_impl::any;
};

paged_selective_ssm_test_params with_state_precision(paged_selective_ssm_test_params params, const ov::element::Type& state_precision) {
    params.state_precision = state_precision;
    return params;
}

paged_selective_ssm_test_params expect_impl(paged_selective_ssm_test_params params, expected_ssm_impl expected) {
    params.expected_impl = expected;
    return params;
}

struct paged_selective_ssm_gpu_test : public ::testing::TestWithParam<paged_selective_ssm_test_params> {
    template <typename DataT, typename StateT>
    static void run_reference(const std::vector<DataT>& A,
                              const std::vector<DataT>& dt,
                              const std::vector<DataT>& B,
                              const std::vector<DataT>& x,
                              const std::vector<DataT>& C,
                              std::vector<StateT>& state,
                              const std::vector<int32_t>& subsequence_begins,
                              const std::vector<int32_t>& block_indices,
                              const std::vector<int32_t>& block_indices_begins,
                              const std::vector<int32_t>& num_processed_tokens,
                              const std::vector<int32_t>& cache_interval,
                              int32_t num_heads,
                              int32_t num_groups,
                              int32_t head_dim,
                              int32_t state_size,
                              std::vector<DataT>& output) {
        const int32_t heads_per_group = num_heads / num_groups;
        const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
        const int32_t tokens = static_cast<int32_t>(x.size()) / (num_heads * head_dim);
        output.resize(static_cast<size_t>(tokens) * num_heads * head_dim);
        const auto state_off = [=](int32_t block, int32_t h, int32_t p, int32_t n) {
            return ((block * num_heads + h) * head_dim + p) * state_size + n;
        };

        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t token_begin = subsequence_begins[seq];
            const int32_t token_end = subsequence_begins[seq + 1];
            const int32_t block_begin = block_indices_begins[seq];
            const int32_t block_end = block_indices_begins[seq + 1];
            const int32_t seq_blocks = std::max(block_end - block_begin, 0);
            const int32_t processed = std::max(num_processed_tokens[seq], 0);
            const int32_t interval = cache_interval[seq];
            const int32_t prev_nums = interval > 0 ? (processed % interval) : 0;
            const int32_t first_block = block_indices[block_begin];

            for (int32_t h = 0; h < num_heads; h++) {
                const int32_t g = h / heads_per_group;
                for (int32_t p = 0; p < head_dim; p++) {
                    std::vector<float> local_state(static_cast<size_t>(state_size), 0.f);
                    for (int32_t n = 0; n < state_size; n++) {
                        local_state[n] = static_cast<float>(state[state_off(first_block, h, p, n)]);
                    }

                    for (int32_t token = token_begin; token < token_end; token++) {
                        const float dt_val = static_cast<float>(dt[token * num_heads + h]);
                        const float dA = std::exp(static_cast<float>(A[h]) * dt_val);
                        const float x_val = static_cast<float>(x[(token * num_heads + h) * head_dim + p]);
                        float acc = 0.f;
                        for (int32_t n = 0; n < state_size; n++) {
                            const float new_state = local_state[n] * dA + x_val * dt_val * static_cast<float>(B[(token * num_groups + g) * state_size + n]);
                            local_state[n] = new_state;
                            acc += local_state[n] * static_cast<float>(C[(token * num_groups + g) * state_size + n]);
                        }
                        output[(token * num_heads + h) * head_dim + p] = static_cast<DataT>(acc);

                        const int32_t processed_now = (token - token_begin) + 1;
                        const int32_t cached_tokens = prev_nums + processed_now;
                        const bool reached_interval_boundary = interval > 0 && ((cached_tokens % interval) == 0);
                        const bool reached_sequence_end = token == token_end - 1;
                        if (reached_interval_boundary || reached_sequence_end) {
                            const int32_t slot = interval > 0 ? 1 + (cached_tokens - 1) / interval : 1;
                            if (slot < seq_blocks) {
                                const int32_t block_id = block_indices[block_begin + slot];
                                for (int32_t n = 0; n < state_size; n++) {
                                    state[state_off(block_id, h, p, n)] = static_cast<StateT>(local_state[n]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename DataT, typename StateT>
    void execute_t(const paged_selective_ssm_test_params& p) {
        auto& engine = get_test_engine();
        const auto data_type = cldnn::element_type_to_data_type(p.precision);
        const auto state_precision = p.state_precision == ov::element::dynamic ? p.precision : p.state_precision;
        const auto state_data_type = cldnn::element_type_to_data_type(state_precision);
        const auto index_data_type = cldnn::element_type_to_data_type(p.index_precision);
        const int32_t num_sequences = static_cast<int32_t>(p.seq_tokens.size());
        const int32_t tokens = static_cast<int32_t>(std::accumulate(p.seq_tokens.begin(), p.seq_tokens.end(), 0));

        std::vector<int32_t> subsequence_begins{0};
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins{0};
        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            subsequence_begins.push_back(subsequence_begins.back() + p.seq_tokens[seq]);
            int32_t required_slots = p.seq_tokens[seq] > 0 ? 2 : 1;
            if (p.cache_intervals[seq] > 0) {
                const int32_t processed = std::max(p.processed_tokens[seq], 0);
                const int32_t prev_nums = processed % p.cache_intervals[seq];
                const int32_t write_blocks = (prev_nums + p.seq_tokens[seq] + p.cache_intervals[seq] - 1) / p.cache_intervals[seq];
                required_slots = 1 + write_blocks;
            }
            std::vector<int32_t> sequence_block_indices(static_cast<size_t>(required_slots));
            for (int32_t i = 0; i < required_slots; i++) {
                sequence_block_indices[i] = total_blocks + (p.reverse_blocks ? required_slots - i - 1 : i);
            }
            if (required_slots > 1 && seq < static_cast<int32_t>(p.alias_first_write.size()) && p.alias_first_write[seq]) {
                sequence_block_indices[1] = sequence_block_indices[0];
            }
            block_indices.insert(block_indices.end(), sequence_block_indices.begin(), sequence_block_indices.end());
            total_blocks += required_slots;
            block_indices_begins.push_back(static_cast<int32_t>(block_indices.size()));
        }
        if (p.invalid_metadata != 0) {
            OPENVINO_ASSERT(num_sequences == 1, "Invalid-metadata tests require one sequence");
            if (p.invalid_metadata == 1) {
                block_indices[0] = total_blocks + 3;
            } else if (p.invalid_metadata == 2) {
                block_indices_begins[1] = static_cast<int32_t>(block_indices.size()) + 1;
            } else {
                OPENVINO_THROW("Unknown invalid metadata mode");
            }
        }

        layout A_layout({p.num_heads}, data_type, format::bfyx);
        layout dt_layout({tokens, p.num_heads}, data_type, format::bfyx);
        layout BC_layout({tokens, p.num_groups, p.state_size}, data_type, format::bfyx);
        layout x_layout({tokens, p.num_heads, p.head_dim}, data_type, format::bfyx);
        layout state_layout({total_blocks, p.num_heads, p.head_dim, p.state_size}, state_data_type, format::bfyx);
        const auto state_memory_layout = p.padded_layouts ? state_layout.with_padding(padding({1, 2, 1, 3}, {2, 1, 2, 1})) : state_layout;
        const auto state_pitches = state_memory_layout.get_pitches();
        const auto state_offset = [&](int32_t block, int32_t h, int32_t dim, int32_t n) {
            return state_memory_layout.get_linear_offset() + static_cast<size_t>(block) * state_pitches[0] + static_cast<size_t>(h) * state_pitches[1] +
                   static_cast<size_t>(dim) * state_pitches[2] + static_cast<size_t>(n) * state_pitches[3];
        };
        layout index_vec_layout({static_cast<int32_t>(subsequence_begins.size())}, index_data_type, format::bfyx);
        layout block_indices_layout({static_cast<int32_t>(block_indices.size())}, index_data_type, format::bfyx);
        layout processed_layout({num_sequences}, index_data_type, format::bfyx);

        // Empty tensors use the runtime's supported dummy-memory representation instead of a zero-byte OpenCL buffer.
        const auto allocate = [&](const layout& requested_layout) {
            if (requested_layout.count() != 0)
                return engine.allocate_memory(requested_layout);
            auto dummy = engine.allocate_memory(layout{{1}, data_types::u8, format::bfyx});
            return engine.reinterpret_buffer(*dummy, requested_layout);
        };

        auto A_mem = allocate(A_layout);
        auto dt_mem = allocate(dt_layout);
        auto B_mem = allocate(BC_layout);
        auto x_mem = allocate(x_layout);
        auto C_mem = allocate(BC_layout);
        auto state_mem = allocate(state_memory_layout);
        auto subseq_mem = allocate(index_vec_layout);
        auto blocks_mem = allocate(block_indices_layout);
        auto block_begins_mem = allocate(index_vec_layout);
        auto processed_mem = allocate(processed_layout);
        auto interval_mem = allocate(processed_layout);

        auto A_data = make_test_values<DataT>(A_mem->count(), -0.03f, -0.2f);
        auto dt_data = make_test_values<DataT>(dt_mem->count(), 0.006f, 0.07f);
        auto B_data = make_test_values<DataT>(B_mem->count(), 0.009f);
        auto x_data = make_test_values<DataT>(x_mem->count(), 0.013f);
        auto C_data = make_test_values<DataT>(C_mem->count(), 0.011f);
        auto state_data = make_test_values<StateT>(state_layout.count(), 0.007f);

        if (p.accumulation_test) {
            A_data.assign(A_data.size(), static_cast<DataT>(0.f));
            dt_data.assign(dt_data.size(), static_cast<DataT>(1.f));
            B_data.assign(B_data.size(), static_cast<DataT>(1.f));
            const float increment = p.precision == ov::element::bf16 ? 0.0546875f : 0.195068359375f;
            x_data.assign(x_data.size(), static_cast<DataT>(increment));
            C_data.assign(C_data.size(), static_cast<DataT>(1.f));
            state_data.assign(state_data.size(), static_cast<StateT>(0.f));
        }

        const auto set_non_empty = [](const memory::ptr& mem, const auto& values) {
            if (!values.empty())
                set_values(mem, values);
        };
        set_non_empty(A_mem, A_data);
        set_non_empty(dt_mem, dt_data);
        set_non_empty(B_mem, B_data);
        set_non_empty(x_mem, x_data);
        set_non_empty(C_mem, C_data);
        if (p.padded_layouts && !state_data.empty()) {
            cldnn::mem_lock<StateT, mem_lock_type::write> state_ptr(state_mem, get_test_stream());
            for (size_t i = 0; i < state_ptr.size(); i++)
                state_ptr[i] = StateT{};
            for (int32_t block = 0; block < total_blocks; block++) {
                for (int32_t h = 0; h < p.num_heads; h++) {
                    for (int32_t dim = 0; dim < p.head_dim; dim++) {
                        for (int32_t n = 0; n < p.state_size; n++) {
                            const auto logical_idx = ((block * p.num_heads + h) * p.head_dim + dim) * p.state_size + n;
                            const auto physical_idx = state_offset(block, h, dim, n);
                            state_ptr[physical_idx] = state_data[logical_idx];
                        }
                    }
                }
            }
        } else {
            set_non_empty(state_mem, state_data);
        }
        if (p.index_precision == ov::element::i64) {
            const std::vector<int64_t> subsequence_begins_i64(subsequence_begins.begin(), subsequence_begins.end());
            const std::vector<int64_t> block_indices_i64(block_indices.begin(), block_indices.end());
            const std::vector<int64_t> block_indices_begins_i64(block_indices_begins.begin(), block_indices_begins.end());
            const std::vector<int64_t> processed_i64(p.processed_tokens.begin(), p.processed_tokens.end());
            const std::vector<int64_t> intervals_i64(p.cache_intervals.begin(), p.cache_intervals.end());
            set_non_empty(subseq_mem, subsequence_begins_i64);
            set_non_empty(blocks_mem, block_indices_i64);
            set_non_empty(block_begins_mem, block_indices_begins_i64);
            set_non_empty(processed_mem, processed_i64);
            set_non_empty(interval_mem, intervals_i64);
        } else {
            set_non_empty(subseq_mem, subsequence_begins);
            set_non_empty(blocks_mem, block_indices);
            set_non_empty(block_begins_mem, block_indices_begins);
            set_non_empty(processed_mem, p.processed_tokens);
            set_non_empty(interval_mem, p.cache_intervals);
        }

        const auto dt_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, p.num_heads}, data_type, format::bfyx} : dt_layout;
        const auto BC_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, p.num_groups, p.state_size}, data_type, format::bfyx} : BC_layout;
        const auto x_input_layout = p.dynamic_shapes ? layout{ov::PartialShape{-1, p.num_heads, p.head_dim}, data_type, format::bfyx} : x_layout;
        const auto state_input_layout =
            p.dynamic_shapes ? layout{ov::PartialShape{-1, p.num_heads, p.head_dim, p.state_size}, state_data_type, format::bfyx} : state_memory_layout;
        const auto dynamic_index_layout = layout{ov::PartialShape{-1}, index_data_type, format::bfyx};
        const auto index_vec_input_layout = p.dynamic_shapes ? dynamic_index_layout : index_vec_layout;
        const auto block_indices_input_layout = p.dynamic_shapes ? dynamic_index_layout : block_indices_layout;
        const auto processed_input_layout = p.dynamic_shapes ? dynamic_index_layout : processed_layout;

        topology topo;
        topo.add(input_layout("A", A_layout));
        topo.add(input_layout("dt", dt_input_layout));
        topo.add(input_layout("B", BC_input_layout));
        topo.add(input_layout("x", x_input_layout));
        topo.add(input_layout("C", BC_input_layout));
        topo.add(input_layout("state", state_input_layout));
        topo.add(input_layout("subsequence_begins", index_vec_input_layout));
        topo.add(input_layout("block_indices", block_indices_input_layout));
        topo.add(input_layout("block_indices_begins", index_vec_input_layout));
        topo.add(input_layout("num_processed_tokens", processed_input_layout));
        topo.add(input_layout("cache_interval", processed_input_layout));
        std::vector<input_info> ssm_inputs{input_info("A"),
                                           input_info("dt"),
                                           input_info("B"),
                                           input_info("x"),
                                           input_info("C"),
                                           input_info("state"),
                                           input_info("subsequence_begins"),
                                           input_info("block_indices"),
                                           input_info("block_indices_begins"),
                                           input_info("num_processed_tokens"),
                                           input_info("cache_interval")};
        if (p.padded_layouts) {
            OPENVINO_ASSERT(!p.dynamic_shapes, "Padded-layout test requires static input layouts");
            const auto A_padding = padding(std::vector<ov::Dimension::value_type>{1}, std::vector<ov::Dimension::value_type>{2});
            const auto metadata_padding = padding(std::vector<ov::Dimension::value_type>{2}, std::vector<ov::Dimension::value_type>{1});
            topo.add(reorder("A_padded", input_info("A"), A_layout.with_padding(A_padding)));
            topo.add(reorder("dt_padded", input_info("dt"), dt_layout.with_padding(padding({1, 2}, {2, 1}))));
            topo.add(reorder("B_padded", input_info("B"), BC_layout.with_padding(padding({1, 2, 3}, {2, 1, 2}))));
            topo.add(reorder("x_padded", input_info("x"), x_layout.with_padding(padding({1, 2, 3}, {2, 1, 2}))));
            topo.add(reorder("C_padded", input_info("C"), BC_layout.with_padding(padding({1, 2, 3}, {2, 1, 2}))));
            topo.add(reorder("subsequence_begins_padded", input_info("subsequence_begins"), index_vec_layout.with_padding(metadata_padding)));
            topo.add(reorder("block_indices_padded", input_info("block_indices"), block_indices_layout.with_padding(metadata_padding)));
            topo.add(reorder("block_indices_begins_padded", input_info("block_indices_begins"), index_vec_layout.with_padding(metadata_padding)));
            topo.add(reorder("num_processed_tokens_padded", input_info("num_processed_tokens"), processed_layout.with_padding(metadata_padding)));
            topo.add(reorder("cache_interval_padded", input_info("cache_interval"), processed_layout.with_padding(metadata_padding)));
            ssm_inputs = {input_info("A_padded"),
                          input_info("dt_padded"),
                          input_info("B_padded"),
                          input_info("x_padded"),
                          input_info("C_padded"),
                          input_info("state"),
                          input_info("subsequence_begins_padded"),
                          input_info("block_indices_padded"),
                          input_info("block_indices_begins_padded"),
                          input_info("num_processed_tokens_padded"),
                          input_info("cache_interval_padded")};
        }
        auto ssm_prim = paged_selective_ssm("paged_selective_ssm", ssm_inputs);
        if (p.padded_layouts)
            ssm_prim.output_paddings = {padding({1, 2, 3}, {2, 1, 2})};
        topo.add(ssm_prim);
        topo.add(reorder("output", input_info("paged_selective_ssm"), format::bfyx, data_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), p.caching_test);
        if (p.expected_impl != expected_ssm_impl::any) {
            const auto primitive = network->get_primitive("paged_selective_ssm");
            ASSERT_NE(primitive, nullptr);
            auto* const impl = primitive->get_impl();
            ASSERT_NE(impl, nullptr);
            const bool is_jit = impl->get_kernel_name().find("jit_") != std::string::npos;
            EXPECT_EQ(is_jit, p.expected_impl == expected_ssm_impl::jit) << "selected implementation: " << impl->get_kernel_name();
        }
        network->set_input_data("A", A_mem);
        network->set_input_data("dt", dt_mem);
        network->set_input_data("B", B_mem);
        network->set_input_data("x", x_mem);
        network->set_input_data("C", C_mem);
        network->set_input_data("state", state_mem);
        network->set_input_data("subsequence_begins", subseq_mem);
        network->set_input_data("block_indices", blocks_mem);
        network->set_input_data("block_indices_begins", block_begins_mem);
        network->set_input_data("num_processed_tokens", processed_mem);
        network->set_input_data("cache_interval", interval_mem);

        std::vector<DataT> ref_output;
        auto ref_state = state_data;
        const auto tolerance_for = [](const ov::element::Type& precision) {
            return precision == ov::element::bf16 ? 0.08f : (precision == ov::element::f16 ? 0.03f : 2e-4f);
        };
        const float output_abs_tolerance = tolerance_for(p.precision);
        const float state_tolerance = tolerance_for(state_precision);
        for (int32_t iteration = 0; iteration < p.iterations; iteration++) {
            auto outputs = network->execute();
            if (p.invalid_metadata != 0) {
                ref_output.assign(static_cast<size_t>(tokens) * p.num_heads * p.head_dim, DataT{});
            } else {
                run_reference(A_data,
                              dt_data,
                              B_data,
                              x_data,
                              C_data,
                              ref_state,
                              subsequence_begins,
                              block_indices,
                              block_indices_begins,
                              p.processed_tokens,
                              p.cache_intervals,
                              p.num_heads,
                              p.num_groups,
                              p.head_dim,
                              p.state_size,
                              ref_output);
            }

            if (!ref_output.empty()) {
                auto out_mem = outputs.at("output").get_memory();
                ASSERT_NE(out_mem, nullptr);
                cldnn::mem_lock<DataT, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
                for (size_t i = 0; i < ref_output.size(); i++) {
                    const float reference = static_cast<float>(ref_output[i]);
                    const float output_tolerance = p.precision == ov::element::f32
                                                       ? std::max(output_abs_tolerance, std::abs(reference) * p.relative_output_tolerance)
                                                       : output_abs_tolerance;
                    ASSERT_NEAR(static_cast<float>(out_ptr[i]), reference, output_tolerance) << "iteration=" << iteration << ", output idx=" << i;
                }
            }

            if (!ref_state.empty()) {
                cldnn::mem_lock<StateT, mem_lock_type::read> state_ptr(state_mem, get_test_stream());
                for (int32_t block = 0; block < total_blocks; block++) {
                    for (int32_t h = 0; h < p.num_heads; h++) {
                        for (int32_t dim = 0; dim < p.head_dim; dim++) {
                            for (int32_t n = 0; n < p.state_size; n++) {
                                const auto logical_idx = ((block * p.num_heads + h) * p.head_dim + dim) * p.state_size + n;
                                const auto physical_idx = state_offset(block, h, dim, n);
                                ASSERT_NEAR(static_cast<float>(state_ptr[physical_idx]), static_cast<float>(ref_state[logical_idx]), state_tolerance)
                                    << "iteration=" << iteration << ", state idx=" << logical_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename DataT>
    void execute_with_data_type(const paged_selective_ssm_test_params& p) {
        const auto state_precision = p.state_precision == ov::element::dynamic ? p.precision : p.state_precision;
        if (state_precision == ov::element::f16) {
            execute_t<DataT, ov::float16>(p);
        } else if (state_precision == ov::element::bf16) {
            execute_t<DataT, ov::bfloat16>(p);
        } else {
            execute_t<DataT, float>(p);
        }
    }

    void execute(const paged_selective_ssm_test_params& p) {
        if (p.precision == ov::element::f16) {
            execute_with_data_type<ov::float16>(p);
        } else if (p.precision == ov::element::bf16) {
            execute_with_data_type<ov::bfloat16>(p);
        } else {
            execute_with_data_type<float>(p);
        }
    }
};

TEST_P(paged_selective_ssm_gpu_test, basic) {
    execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    smoke_paged_selective_ssm_gpu_test,
    paged_selective_ssm_gpu_test,
    ::testing::Values(
        paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{2, 4, 1}, {1, 2, 3}, {1, 3, 2}, 4, 1, 8, 16, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::f16, ov::element::i32, false},
        paged_selective_ssm_test_params{{2, 1}, {0, 3}, {2, 1}, 2, 1, 4, 16, ov::element::f32, ov::element::i64, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 16, ov::element::bf16, ov::element::i32, false},
        // The data path and recurrent state have independent floating-point types. A state size below the
        // subgroup width exercises the universal optimized path without depending on its internal kernel name.
        with_state_precision(paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::f32, ov::element::i32, false}, ov::element::bf16),
        with_state_precision(paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::bf16, ov::element::i32, false}, ov::element::f16),
        expect_impl(paged_selective_ssm_test_params{{8}, {0}, {8}, 64, 1, 64, 128, ov::element::f32, ov::element::i32, false}, expected_ssm_impl::jit),
        expect_impl(with_state_precision(paged_selective_ssm_test_params{{8}, {0}, {8}, 64, 1, 64, 128, ov::element::f32, ov::element::i32, false},
                                         ov::element::f16),
                    expected_ssm_impl::jit),
        // Non-prefix-cached GenAI decode uses interval zero and aliases the read block with the final live-state write.
        // Multiple executions ensure that every decode step consumes the state produced by the preceding one.
        expect_impl(
            with_state_precision(
                paged_selective_ssm_test_params{{1}, {0}, {0}, 64, 1, 64, 128, ov::element::f32, ov::element::i32, true, {true}, false, false, false, 4},
                ov::element::f16),
            expected_ssm_impl::jit),
        paged_selective_ssm_test_params{{8}, {0}, {8}, 64, 1, 64, 128, ov::element::f16, ov::element::i32, false},
        expect_impl(paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 256, ov::element::f32, ov::element::i32, true, {}, false, true},
                    expected_ssm_impl::jit),
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 256, ov::element::f16, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 256, ov::element::bf16, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 512, ov::element::f16, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 512, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 512, ov::element::bf16, ov::element::i32, false},
        expect_impl(paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 2, 513, ov::element::f32, ov::element::i32, false}, expected_ssm_impl::fallback),
        paged_selective_ssm_test_params{{2, 3}, {1, 0}, {2, 3}, 4, 2, 4, 16, ov::element::f32, ov::element::i64, true},
        // Five page-aliasing cases from the published specification.
        paged_selective_ssm_test_params{{5}, {0}, {2}, 4, 2, 5, 17, ov::element::f32, ov::element::i32, false, {true}, true},
        paged_selective_ssm_test_params{{5}, {4}, {2}, 4, 2, 5, 17, ov::element::f32, ov::element::i32, false, {false}, true},
        paged_selective_ssm_test_params{{5}, {3}, {2}, 4, 2, 5, 17, ov::element::f16, ov::element::i32, false, {true}, true},
        paged_selective_ssm_test_params{{1}, {4}, {2}, 4, 2, 5, 17, ov::element::f32, ov::element::i64, false, {false}, true},
        paged_selective_ssm_test_params{{1}, {3}, {2}, 4, 2, 5, 17, ov::element::bf16, ov::element::i64, false, {true}, true},
        // Interval zero disables intermediate checkpoints but still persists the final live state in slot one.
        paged_selective_ssm_test_params{{4, 3}, {9, 2}, {0, -3}, 4, 2, 4, 31, ov::element::f32, ov::element::i64, false, {true, true}, true},
        // Empty sequences and completely empty token batches are legal no-ops.
        paged_selective_ssm_test_params{{0, 3, 0}, {0, 1, 7}, {2, 2, 0}, 4, 2, 4, 16, ov::element::f32, ov::element::i32, false, {true, false, false}, true},
        paged_selective_ssm_test_params{{0}, {0}, {2}, 2, 1, 2, 8, ov::element::f32, ov::element::i32, true, {true}, false},
        // Exercise binary serialization of the optimized implementation.
        paged_selective_ssm_test_params{{3, 2}, {1, 0}, {2, 2}, 4, 2, 4, 16, ov::element::f16, ov::element::i64, true, {true, true}, true, true},
        expect_impl(
            paged_selective_ssm_test_params{{4, 2}, {3, 7}, {2, 3}, 4, 2, 3, 19, ov::element::f32, ov::element::i64, false, {true, false}, true, false, true},
            expected_ssm_impl::fallback),
        paged_selective_ssm_test_params{{3, 2}, {1, 0}, {2, 2}, 4, 2, 4, 16, ov::element::f16, ov::element::i32, true, {true, true}, true, false, false, 3},
        with_state_precision(
            paged_selective_ssm_test_params{{3, 2}, {1, 0}, {2, 2}, 4, 2, 4, 16, ov::element::f16, ov::element::i32, true, {true, true}, true, false, false, 3},
            ov::element::f32),
        paged_selective_ssm_test_params{{32, 17, 5}, {3, 7, 1}, {4, 3, 2}, 8, 4, 16, 64, ov::element::f16, ov::element::i64, true},
        // Exercise serialization of a dynamic-shape configuration eligible for the device-specific JIT kernels.
        paged_selective_ssm_test_params{{32, 17, 5}, {3, 7, 1}, {4, 3, 2}, 8, 4, 16, 64, ov::element::f16, ov::element::i64, true, {}, false, true},
        // Exercise serialization of the long dynamic recurrence that selects Xe2 dA precomputation.
        expect_impl(paged_selective_ssm_test_params{{3072}, {0}, {3072}, 2, 1, 64, 128, ov::element::f16, ov::element::i32, true, {}, false, true},
                    expected_ssm_impl::jit),
        paged_selective_ssm_test_params{{128}, {0}, {32}, 4, 2, 8, 32, ov::element::f16, ov::element::i32, false},
        paged_selective_ssm_test_params{{128}, {0}, {32}, 4, 2, 8, 32, ov::element::bf16, ov::element::i32, false},
        paged_selective_ssm_test_params{{128}, {0}, {128}, 1, 1, 1, 1, ov::element::f16, ov::element::i32, false, {}, false, false, false, 1, 0, true},
        paged_selective_ssm_test_params{{128}, {0}, {128}, 1, 1, 1, 1, ov::element::bf16, ov::element::i32, false, {}, false, false, false, 1, 0, true},
        paged_selective_ssm_test_params{{3}, {-7}, {2}, 2, 1, 4, 8, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{3}, {0}, {2}, 2, 1, 4, 8, ov::element::f32, ov::element::i32, false, {}, false, false, false, 1, 1},
        paged_selective_ssm_test_params{{3}, {0}, {2}, 2, 1, 4, 8, ov::element::f16, ov::element::i64, false, {}, false, false, false, 1, 2},
        expect_impl(paged_selective_ssm_test_params{{3}, {0}, {2}, 2, 1, 4, 64, ov::element::f32, ov::element::i32, false, {}, false, false, false, 1, 1},
                    expected_ssm_impl::jit),
        expect_impl(paged_selective_ssm_test_params{{3}, {0}, {2}, 2, 1, 4, 64, ov::element::f16, ov::element::i64, false, {}, false, false, false, 1, 2},
                    expected_ssm_impl::jit),
        expect_impl(with_state_precision(
                        paged_selective_ssm_test_params{{3}, {0}, {2}, 2, 1, 4, 64, ov::element::f16, ov::element::i64, false, {}, false, false, false, 2},
                        ov::element::bf16),
                    expected_ssm_impl::jit),
        // Exercise local-memory-driven 4 -> 3 -> 2 -> 1 blocking and tails.
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 5000, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 3, 6000, ov::element::f32, ov::element::i32, false},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 2, 1, 4, 8192, ov::element::f32, ov::element::i32, false},
        // Exceed 128 KiB of FP32 state and exercise the universal global-state fallback.
        paged_selective_ssm_test_params{{2},
                                        {0},
                                        {2},
                                        1,
                                        1,
                                        1,
                                        large_state_size,
                                        ov::element::f32,
                                        ov::element::i32,
                                        false,
                                        {},
                                        false,
                                        false,
                                        true,
                                        1,
                                        0,
                                        false,
                                        2e-4f},
        paged_selective_ssm_test_params{{2}, {0}, {2}, 1, 1, 1, large_state_size, ov::element::f16, ov::element::i32, true, {}, false, true},
        with_state_precision(
            paged_selective_ssm_test_params{{2}, {0}, {2}, 1, 1, 1, large_state_size, ov::element::f16, ov::element::i32, true, {}, false, true},
            ov::element::bf16)));

}  // namespace
