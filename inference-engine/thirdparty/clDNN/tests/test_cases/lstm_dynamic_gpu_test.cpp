// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/mutable_data.hpp>
#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/lstm.hpp>
#include <cldnn/primitives/lstm_dynamic.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/lstm_dynamic_input.hpp>
#include <cldnn/primitives/lstm_dynamic_timeloop.hpp>

#include <chrono>
#include <sstream>
#include <iomanip>

#ifndef __clang__
#pragma warning( disable : 4503 )
#endif

#define MEASURE_PERF false
#define MEASURE_LOOP 50
using namespace cldnn;
using namespace ::tests;

namespace {
    float sigmoid(float x) {
        return 1.f / (1.f + (float)std::exp((float)(-x)));
    }
}

struct offset_order_dynamic {
    size_t it, ot, ft, zt;
    offset_order_dynamic(size_t scale, const lstm_weights_order& t = lstm_weights_order::fizo) {
        static const std::map<lstm_weights_order, std::vector<size_t>> offset_map{
            { lstm_weights_order::fizo, { 1, 3, 0, 2 } },
        };
        std::vector<size_t> v = offset_map.at(t);
        it = v[0] * scale;
        ot = v[1] * scale;
        ft = v[2] * scale;
        zt = v[3] * scale;
    }
};
lstm_weights_order default_offset_type_dynamic = lstm_weights_order::fizo;

namespace dynamic_lstm
{
    template<typename T>
    T clip(T val, T threshold) {
        if (threshold > 0) {
            if (val > threshold) return threshold;
            if (val < -threshold) return -threshold;
        }
        return val;
    }

template <typename T>
VVVVF<T> lstm_dynamic_input_ref(VVVVF<T>& input, VVVVF<T>& weights, VVVVF<T>& bias,
    VF<float> dynamic_lengths, size_t seq, bool hasBias, size_t dir) {
    size_t input_size = input[0][0][0].size();
    size_t hidden_size = weights[0][0].size() / 4;
    size_t batch_size = input.size();

    VVVVF<T>output(batch_size, VVVF<T>(seq, VVF<T>(dir, VF<T>(4 * hidden_size))));
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t l = 0; l < seq; ++l)
        {
            if (l > static_cast<size_t>(dynamic_lengths[b]))
                break;
            for (size_t d = 0; d < dir; ++d)
            {
                for (size_t y = 0; y < 4 * hidden_size; ++y)
                {
                    T res = 0;
                    for (size_t x = 0; x < input_size; ++x)
                    {
                        res += (T)weights[0][d][y][x] * (T)input[b][l][d][x];
                    }
                    if (hasBias)
                    {
                        res += (T)bias[0][0][d][y];
                    }
                    output[b][l][d][y] = res;
                }
            }
        }
    }
    return output;
}

    template <typename T>
    VVVVF<T> lstm_gemm_reference(VVVVF<T>& input, VVVVF<T>& weights, VVVVF<T>& recurrent, VVVVF<T>& bias, VVVVF<T>& hidden,
        size_t seq, bool hasBias = true, bool hasHidden = true, size_t dir = 0, size_t input_dir = 0) {
        size_t input_size = input[0][0][0].size();
        size_t hidden_size = hidden[0][0][0].size();
        size_t batch_size = input.size();

        // Temporary output from GEMM operations [f, i, o, z]
        VVVVF<T> tempGEMM(batch_size, VVVF<T>(1, VVF<T>(1, VF<T>(4 * hidden_size))));
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t y = 0; y < 4 * hidden_size; ++y) {
                T res = 0;
                for (size_t x = 0; x < input_size; ++x) {
                    res += (T)weights[0][dir][y][x] * (T)input[b][seq][input_dir][x];
                }
                if (hasHidden) {
                    for (size_t x = 0; x < hidden_size; ++x) {
                        auto rec_v = (T)recurrent[0][dir][y][x];
                        auto hid_v = (T)hidden[b][0][dir][x];
                        auto temp = rec_v * hid_v;
                        res += temp;
                    }
                }
                if (hasBias) {
                    res += (T)bias[0][0][dir][y];
                }
                tempGEMM[b][0][0][y] = res;
            }
        }
        return tempGEMM;
    }

    template <typename T>
    VVVVF<T> lstm_elt_reference(VVVVF<T>& tempGEMM, VVVVF<T>& cell,
        bool hasCell = true, float clip_threshold = 0,
        bool input_forget = false, size_t dir = 0)
    {
        size_t hidden_size = tempGEMM[0][0][0].size() / 4;
        size_t batch_size = tempGEMM.size();
        VVVVF<T> tempOut(batch_size, VVVF<T>(2, VVF<T>(1, VF<T>(hidden_size))));
        offset_order_dynamic off(hidden_size, default_offset_type_dynamic);

        for (size_t b = 0; b < batch_size; ++b) {
            T *it = &tempGEMM[b][0][0][off.it];
            T *ot = &tempGEMM[b][0][0][off.ot];
            T *ft = &tempGEMM[b][0][0][off.ft];
            T *zt = &tempGEMM[b][0][0][off.zt];

            for (size_t h = 0; h < hidden_size; ++h) {

                // Convert all inputs to float for all the elementwise operations. This is done to immitate
                // how lstm kernel is performing the elementwise operations.
                float fp32_it = (float)it[h];
                float fp32_ot = (float)ot[h];
                float fp32_ft = (float)ft[h];
                float fp32_zt = (float)zt[h];
                float val = sigmoid(clip(fp32_it, clip_threshold)) * std::tanh(clip(fp32_zt, clip_threshold));

                if (input_forget) {
                    val *= (1 - fp32_ft);
                }
                if (hasCell) {
                    val += (float)cell[b][0][dir][h] * sigmoid(clip(fp32_ft, clip_threshold));
                }

                // Convert back to output data type before storing it into the output buffer. Currently, the output
                // data type may be float or FLOAT16 (half)
                tempOut[b][0][0][h] = (T)(std::tanh(val) * sigmoid(fp32_ot));
                tempOut[b][1][0][h] = (T)val;
            }
        }
        return tempOut;
    }

    template <typename T>
    void lstm_dynamic_reference(VVVVF<T>& input, VVVVF<T>& hidden, VVVVF<T>& cell,
        VVVVF<T>& weights, VVVVF<T>& recurrent, VVVVF<T>& bias,
        VVVVF<T>& output_hidden, VVVVF<T>& output_cell,
        bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true,
        float clip_threshold = 0, bool input_forget = false)
    {
        size_t sequence_len = input[0].size();
        size_t dir_len = weights[0].size();
        size_t batch = input.size();
        for (size_t dir = 0; dir < dir_len; ++dir) {
            bool tempHasInitialHidden = hasInitialHidden;
            bool tempHasInitialCell = hasInitialCell;
            for (size_t seq = 0; seq < sequence_len; ++seq) {
                size_t seq_id = seq;
                size_t input_direction = dir;
                VVVVF<T> tempGEMM = lstm_gemm_reference(input, weights, recurrent, bias, hidden, seq_id, hasBias, tempHasInitialHidden, dir, input_direction);
                VVVVF<T> tempOutput = lstm_elt_reference(tempGEMM, cell, tempHasInitialCell, clip_threshold, input_forget, dir);
                // tempOutput[batch][0] = hidden and tempOutput[batch][1] = cell
                for (size_t i = 0; i < batch; i++) {
                    output_hidden[i][seq][dir] = tempOutput[i][0][0];
                    output_cell[i][seq][dir] = tempOutput[i][1][0];
                    hidden[i][0][dir] = tempOutput[i][0][0];
                    cell[i][0][dir] = tempOutput[i][1][0];
                }
                tempHasInitialHidden = true;
                tempHasInitialCell = true;
            }
        }
    }
}
template <typename T>
struct lstm_dynamic_input_layer_test : public ::testing::Test
{
    void input_single_layer_generic_test(int32_t direction, int32_t batch_size, int32_t max_sequence_len, int32_t input_size, int32_t hidden_size, std::vector<float> dynamic_lengths,
        bool has_bias = false)
    {
        auto min_random = -2, max_random = 2;
        VVVVF<T> ref_input = generate_random_4d<T>(batch_size, max_sequence_len, direction, input_size, min_random, max_random);
        VVVVF<T> ref_weights = generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
        VVVVF<T> ref_bias = generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);

        VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
        VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
        VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);

        auto& engine = get_test_engine();
        VF<T> ref_dynamic_length;
        for (auto& v : dynamic_lengths)
            ref_dynamic_length.push_back((T)v);
        constexpr auto dt = std::is_same<T, float>::value ? data_types::f32 : data_types::f16;

        auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, direction } });
        set_values<T>(input_mem, ref_input_vec);
        auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
        set_values<T>(weights_mem, ref_weights_vec);
        auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, batch_size, 1 } });
        set_values<T>(dynamic_length_mem, ref_dynamic_length);
        auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, direction } });
        set_values(bias_mem, ref_bias_vec);

        topology topology;
        topology.add(input_layout("input", input_mem->get_layout()));
        topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
        topology.add(data("weights", weights_mem));

        std::string bias_id = "";
        if (has_bias) {
            bias_id = "bias";
            topology.add(data(bias_id, bias_mem));
        }

        topology.add(lstm_dynamic_input("dynamic_lstm_input",
            "input",
            "dyn_len",
            "weights",
            bias_id));

        build_options opts;
        opts.set_option(build_option::optimize_data(true));
        network network(engine, topology, opts);

#if MEASURE_PERF == true
        using clock = std::chrono::high_resolution_clock;
        std::vector<std::chrono::nanoseconds> times(MEASURE_LOOP);
        for (uint32_t i = 0; i < MEASURE_LOOP; i++)
        {
            auto t0 = clock::now();
            network.set_input_data("input", input_mem);
            network.set_input_data("dynamic_lstm_input", dynamic_length_mem);
            auto real_outs = network.execute();
            real_outs.at("dynamic_lstm_input").get_event().wait();
            auto t1 = clock::now();
            auto exec_time = t1 - t0;
            times[i] = exec_time;
        }
        std::sort(times.begin(), times.end());
        std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
        std::cout << "Perf: " << std::chrono::duration_cast<std::chrono::microseconds>(times[times.size() / 2]).count() << " micros. " << std::endl;
#else
        network.set_input_data("input", input_mem);
        network.set_input_data("dyn_len", dynamic_length_mem);

        auto outputs = network.execute();
        auto out = outputs.at("dynamic_lstm_input");
        auto out_tensor = out.get_memory()->get_layout().size;
        cldnn::mem_lock<T> out_ptr(out.get_memory(), get_test_stream());


        auto output_ref =  dynamic_lstm::lstm_dynamic_input_ref(ref_input, ref_weights, ref_bias, dynamic_lengths, max_sequence_len, has_bias, direction);

        size_t i = 0;
        for (auto b = 0; b < out_tensor.batch[0]; b++)
        {
            for (auto len = 0; len < max_sequence_len; len++)
            {
                for (auto dir = 0; dir < direction; dir++)
                {
                    for (auto x = 0; x < out_tensor.spatial[0]; x++)
                    {
                        EXPECT_NEAR(output_ref[b][len][dir][x], (float)out_ptr[i++], 1e-3f)
                            << "b:" << b << ", "
                            << "len:" << len << ", "
                            << "dir:" << dir << ", "
                            << "x:" << x << ", "
                            << std::endl;
                    }
                }
            }
        }
#endif
    }
};

template <typename T>
struct lstm_dynamic_single_layer_test : public ::testing::Test
{
    void single_layer_generic_test(int32_t direction, int32_t batch_size, int32_t max_sequence_len, int32_t input_size, int32_t hidden_size, std::vector<float> dynamic_lengths,
        bool has_bias = false, bool has_initial_hidden = false, bool has_initial_cell = false, bool has_last_hidden_state = false, bool has_last_cell_state = false, float epsilon = 1e-3f)
    {
        float clip_threshold = 0;
        bool input_forget = false;

        auto min_random = 0, max_random = 2;
        VVVVF<T> ref_input = generate_random_4d<T>(batch_size, max_sequence_len, direction, input_size, min_random, max_random);
        VVVVF<T> ref_weights = generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
        VVVVF<T> ref_recurrent = generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random);
        VVVVF<T> ref_bias = generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);
        VVVVF<T> ref_hidden = generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random);
        VVVVF<T> ref_cell = generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random);
        VVVVF<T> ref_output_hidden = VVVVF<T>(batch_size, VVVF<T>(max_sequence_len, VVF<T>(direction, VF<T>(hidden_size))));
        VVVVF<T> ref_output_cell = VVVVF<T>(batch_size, VVVF<T>(max_sequence_len, VVF<T>(direction, VF<T>(hidden_size))));

        VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
        VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
        VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
        VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
        VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);
        VF<T> ref_cell_vec = flatten_4d<T>(cldnn::format::bfyx, ref_cell);

        auto& engine = get_test_engine();
        constexpr auto dt = std::is_same<T, float>::value ? data_types::f32 : data_types::f16;
        VF<T> ref_dynamic_length;
        for (auto& v : dynamic_lengths)
            ref_dynamic_length.push_back((T)v);

        auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, direction } });
        set_values<T>(input_mem, ref_input_vec);

        auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
        set_values<T>(weights_mem, ref_weights_vec);
        auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, hidden_size, 4 * hidden_size } });
        set_values<T>(recurrent_mem, ref_recurrent_vec);
        auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, batch_size, 1 } });
        set_values<T>(dynamic_length_mem, ref_dynamic_length);
        auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, direction } });
        set_values(bias_mem, ref_bias_vec);
        auto initial_hidden_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, 1, hidden_size, direction } });
        set_values<T>(initial_hidden_mem, ref_hidden_vec);
        auto initial_cell_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, 1, hidden_size, direction } });
        set_values<T>(initial_cell_mem, ref_cell_vec);

        topology topology;
        topology.add(input_layout("input", input_mem->get_layout()));
        topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
        topology.add(data("weights", weights_mem));
        topology.add(data("recurrent", recurrent_mem));

        std::string bias_id = "";
        if (has_bias)
        {
            bias_id = "bias";
            topology.add(data(bias_id, bias_mem));
        }

        std::string initial_hidden_id = "";
        if (has_initial_hidden)
        {
            initial_hidden_id = "initial_hidden";
            topology.add(data(initial_hidden_id, initial_hidden_mem));
        }

        std::string initial_cell_id = "";
        if (has_initial_cell)
        {
            initial_cell_id = "initial_cell";
            topology.add(data(initial_cell_id, initial_cell_mem));
        }

        std::string last_hidden_state = "";
        auto last_hidden_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, 1, hidden_size, direction } });
        last_hidden_mem->fill(get_test_stream());
        get_test_stream().finish();
        if (has_last_hidden_state)
        {
            last_hidden_state = "last_hidden_state";
            topology.add(mutable_data(last_hidden_state, last_hidden_mem));
        }

        std::string last_cell_state = "";
        auto last_cell_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, 1, hidden_size, direction } });
        last_cell_mem->fill(get_test_stream());
        get_test_stream().finish();
        if (has_last_cell_state)
        {
            last_cell_state = "last_cell_state";
            topology.add(mutable_data(last_cell_state, last_cell_mem));
        }

        topology.add(lstm_dynamic("dynamic_lstm",
            "input",
            "dyn_len",
            "weights",
            "recurrent",
            last_hidden_state,
            last_cell_state,
            bias_id,
            initial_hidden_id,
            initial_cell_id));

        build_options opts;
        opts.set_option(build_option::optimize_data(true));
        network network(engine, topology, opts);
        network.set_input_data("input", input_mem);
        network.set_input_data("dyn_len", dynamic_length_mem);

#if MEASURE_PERF == true
        using clock = std::chrono::high_resolution_clock;
        std::vector<std::chrono::nanoseconds> times(MEASURE_LOOP);
        for (uint32_t i = 0; i < MEASURE_LOOP; i++)
        {
            auto t0 = clock::now();
            network.set_input_data("input", input_mem);
            network.set_input_data("dyn_len", dynamic_length_mem);
            auto real_outs = network.execute();
            real_outs.at("dynamic_lstm").get_event().wait();
            auto t1 = clock::now();
            auto exec_time = t1 - t0;
            times[i] = exec_time;
        }
        std::sort(times.begin(), times.end());
        std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
        std::cout << "Perf: " << std::chrono::duration_cast<std::chrono::microseconds>(times[times.size() / 2]).count() << " micros. " << std::endl;
#else
        dynamic_lstm::lstm_dynamic_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output_hidden,
            ref_output_cell, has_bias, has_initial_hidden, has_initial_cell,
            clip_threshold, input_forget);
        auto real_outs = network.execute();
        auto out = real_outs.at("dynamic_lstm");
        auto out_tensor = out.get_memory()->get_layout().size;

        cldnn::mem_lock<T> out_ptr(out.get_memory(), get_test_stream());
        cldnn::mem_lock<T> last_hidden_ptr(last_hidden_mem, get_test_stream());
        cldnn::mem_lock<T> last_cell_ptr(last_cell_mem, get_test_stream());
        size_t i = 0, i_lh = 0, i_lc = 0;
        for (auto b = 0; b < out_tensor.batch[0]; b++)
        {
            for (auto len = 0; len < max_sequence_len; len++)
            {
                for (auto dir = 0; dir < direction; dir++)
                {
                    for (auto x = 0; x < out_tensor.spatial[0]; x++)
                    {
                        //check hidden
                        if (len < dynamic_lengths[b])
                        {
                            EXPECT_NEAR((float)ref_output_hidden[b][len][dir][x], (float)out_ptr[i++], epsilon)
                                << "check hidden, "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;
                        }
                        else
                        {
                            EXPECT_NEAR(0.0f, (float)out_ptr[i++], epsilon)
                                << "check hidden, "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;
                        }

                        //check optional last hidden state output
                        if(has_last_hidden_state && len == dynamic_lengths[b] - 1)
                        {
                            auto ratio = (float)ref_output_hidden[b][len][dir][x] / (float)last_hidden_ptr[i_lh++];
                            EXPECT_TRUE(std::abs(1.0f - ratio) < 0.01f)
                            << "check has_last_hidden_state with ratio: " << ratio << ", "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;

                        }
                        else if (has_last_hidden_state && len == 0 && dynamic_lengths[b] == 0)
                        {
                            EXPECT_NEAR(0.0f, (float)last_hidden_ptr[i_lh++], epsilon)
                                << "check has_last_hidden_state, "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;
                        }

                        //check optional last cell state output
                        if(has_last_cell_state && len == dynamic_lengths[b] - 1)
                        {
                            auto ratio = (float)ref_output_cell[b][len][dir][x] / (float)last_cell_ptr[i_lc++];
                            EXPECT_TRUE(std::abs(1.0f - ratio) < 0.01f)
                                << "check has_last_cell_state with ratio: " << ratio << ", "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;
                        }
                        else if (has_last_cell_state && len == 0 && dynamic_lengths[b] == 0)
                        {
                            EXPECT_NEAR(0.0f, (float)last_cell_ptr[i_lc++], epsilon)
                                << "check has_last_cell_state, "
                                << "b:" << b << ", "
                                << "len:" << len << ", "
                                << "dir:" << dir << ", "
                                << "x:" << x << ", "
                                << std::endl;
                        }
                    }
                }
            }
        }
#endif
    }

};
typedef ::testing::Types<float, FLOAT16> lstm_dynamic_test_types;
TYPED_TEST_SUITE(lstm_dynamic_single_layer_test, lstm_dynamic_test_types);
TYPED_TEST_SUITE(lstm_dynamic_input_layer_test, lstm_dynamic_test_types);

/*
----------------------------------------------
        DYNAMIC_LSTM INPUT TEST
----------------------------------------------
*/

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_b1_seq3_is3_hs2)
{
    auto dir = 1, batch_size = 1, max_seq_len = 5, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 3 };
    this->input_single_layer_generic_test(dir, batch_size, max_seq_len, input_size, hidden_size, dynamic_lengths, true);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_b3_seq5_is3_hs2)
{
    auto dir = 1, batch_size = 3, max_seq_len = 5, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 3, 4, 2 };
    this->input_single_layer_generic_test(dir, batch_size, max_seq_len, input_size, hidden_size, dynamic_lengths, true);
}

TYPED_TEST(lstm_dynamic_input_layer_test, b10_seq20_is16_hs64)
{
    auto dir = 1, batch = 10, max_seq_len = 20, input_size = 16, hidden_size = 64;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8, 9, 15,
    };
    this->input_single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_b8_seq10_is4_hs16)
{
    auto batch_size = 8, max_seq_len = 10, input_size = 4, hidden_size = 16;
    std::vector<float> dynamic_lengths = { 1, 2, 3, 4, 5, 6, 7, 8};
    auto dir = 1;
    this->input_single_layer_generic_test(dir, batch_size, max_seq_len, input_size, hidden_size, dynamic_lengths, true);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_dir2_b8_seq10_is4_hs16_options)
{
    auto batch_size = 8, max_seq_len = 10, input_size = 4, hidden_size = 16;
    std::vector<float> dynamic_lengths = { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto dir = 2;
    std::vector<bool> bias_options = { true, false };
    for (auto bias : bias_options)
    {
        this->input_single_layer_generic_test(dir, batch_size, max_seq_len, input_size, hidden_size, dynamic_lengths, bias);
    }
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_1b1_seq1_is32_hs_128)
{
    auto dir = 1, batch = 1, max_seq_len = 1, input_size = 32, hidden_size = 128;
    std::vector<float> dynamic_lengths =
    {
        1
    };
    bool bias = true;
    this->input_single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_dir_b8_seq27_is16_hs_56)
{
    auto dir = 1, batch = 8, max_seq_len = 27, input_size = 16, hidden_size = 56;
    std::vector<float> dynamic_lengths =
    {
        20, 25, 24, 10, 15, 8, 19, 26
    };
    this->input_single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, false);
}


/*
----------------------------------------------
        FULL DYNAMIC_LSTM TESTS
----------------------------------------------
*/

TYPED_TEST(lstm_dynamic_single_layer_test, b1_seq1_is3_hs2)
{
    auto dir = 1, batch = 1, max_seq_len = 1, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 1 };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, b1_seq3_is3_hs2_options)
{
    auto dir = 1, batch = 1, max_seq_len = 3, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 1 };
    std::vector<bool> bias_options = { true, false };
    std::vector<bool> init_hidden = { true, false };
    std::vector<bool> init_cell = { true, false };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c);
            }
        }
    }
}

TYPED_TEST(lstm_dynamic_single_layer_test, b1_seq10_is10_hs32)
{
    auto dir = 1, batch = 1, max_seq_len = 10, input_size = 10, hidden_size = 32;
    std::vector<float> dynamic_lengths = { 8 };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, b1_seq10_is10_hs32_options)
{
    auto dir = 1, batch = 1, max_seq_len = 10, input_size = 10, hidden_size = 32;
    std::vector<float> dynamic_lengths = { 8 };
    std::vector<bool> bias_options = { true, false };
    std::vector<bool> init_hidden = { true, false };
    std::vector<bool> init_cell = { true, false };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c);
            }
        }
    }
}

TYPED_TEST(lstm_dynamic_single_layer_test, b4_seq1_is3_hs2)
{
    auto dir = 1, batch = 2, max_seq_len = 3, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 1, 2 };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, b4_seq3_is3_hs2_options)
{
    auto dir = 1, batch = 4, max_seq_len = 3, input_size = 3, hidden_size = 2;
    std::vector<float> dynamic_lengths = { 1, 2, 2, 0 };
    std::vector<bool> bias_options = { true, false };
    std::vector<bool> init_hidden = { true, false };
    std::vector<bool> init_cell = { true, false };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c);
            }
        }
    }
}

TYPED_TEST(lstm_dynamic_single_layer_test, b10_seq20_is16_hs64)
{
    auto dir = 1, batch = 10, max_seq_len = 20, input_size = 16, hidden_size = 64;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8, 9, 15,
    };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

// DISABLED beacuse it is veeery long
TYPED_TEST(lstm_dynamic_single_layer_test, DISABLED_b16_seq20_is32_hs32_options)
{
    auto dir = 1, batch = 16, max_seq_len = 20, input_size = 32, hidden_size = 32;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8,  9,  15, 0, 0, 0, 0, 19, 18
    };
    std::vector<bool> bias_options = { true, false };
    std::vector<bool> init_hidden = { true, false };
    std::vector<bool> init_cell = { true, false };
    std::vector<bool> last_hidden_state = { true, false };
    std::vector<bool> last_cell_state = { true, false };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                for (auto l_h_s : last_hidden_state)
                {
                    for (auto l_c_s : last_cell_state)
                    {
                        this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c, l_h_s, l_c_s, 1e-2f);
                    }
                }
            }
        }
    }
}

/*
----------------------------------------------
              BIDIRECTIONAL TESTS
----------------------------------------------
*/

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b2_seq7_is3_hs4)
{
    auto dir = 2, batch = 2, max_seq_len = 7, input_size = 3, hidden_size = 4;
    std::vector<float> dynamic_lengths = { 3, 5 };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_dir_b1_seq1_is32_hs_512)
{
    auto dir = 2, batch = 1, max_seq_len = 1, input_size = 8, hidden_size = 128;
    std::vector<float> dynamic_lengths =
    {
        1
    };
    this->input_single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, true);
}

TYPED_TEST(lstm_dynamic_input_layer_test, dlstm_input_dir_b8_seq5_is32_hs_512)
{
    auto dir = 2, batch = 8, max_seq_len = 5, input_size = 8, hidden_size = 128;
    std::vector<float> dynamic_lengths =
    {
        3, 4, 5, 1, 3, 2, 2, 3
    };
    this->input_single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, true);
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b10_seq7_is3_hs4)
{
    auto dir = 2, batch = 10, max_seq_len = 7, input_size = 3, hidden_size = 4;
    std::vector<float> dynamic_lengths = { 1, 2, 3, 4, 5, 6, 5, 4, 3, 2};
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b2_seq7_is3_hs4_options)
{
    auto dir = 2, batch = 2, max_seq_len = 7, input_size = 3, hidden_size = 4;
    std::vector<float> dynamic_lengths = { 3, 5 };
    std::vector<bool> bias_options = { false, true };
    std::vector<bool> init_hidden = { false, true };
    std::vector<bool> init_cell = { false, true};
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c);
            }
        }
    }
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b1_seq10_is10_hs32)
{
    auto dir = 2, batch = 1, max_seq_len = 10, input_size = 10, hidden_size = 32;
    std::vector<float> dynamic_lengths = { 8 };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b1_seq10_is10_hs32_options)
{
    auto dir = 2, batch = 1, max_seq_len = 10, input_size = 10, hidden_size = 32;
    std::vector<float> dynamic_lengths = { 8 };
    std::vector<bool> bias_options = { true, false };
    std::vector<bool> init_hidden = { true, false };
    std::vector<bool> init_cell = { true, false };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c, false, false, 1e-2f);
            }
        }
    }
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b10_seq20_is16_hs64)
{
    auto dir = 2, batch = 10, max_seq_len = 20, input_size = 16, hidden_size = 64;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8,  9,  15,
    };
    this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths);
}

TYPED_TEST(lstm_dynamic_single_layer_test, bidir_b16_seq20_is4_hs8_options)
{
    auto dir = 2, batch = 16, max_seq_len = 20, input_size = 4, hidden_size = 8;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8, 9, 15, 0, 0, 0, 0, 14, 18
    };
    std::vector<bool> bias_options = { false, true };
    std::vector<bool> init_hidden = { false, true };
    std::vector<bool> init_cell = { false, true };
    for (auto bias : bias_options)
    {
        for (auto i_h : init_hidden)
        {
            for (auto i_c : init_cell)
            {
                this->single_layer_generic_test(dir, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, bias, i_h, i_c);
            }
        }
    }
}

/*
----------------------------------------------
                OPTIONAL OUTPUTS
----------------------------------------------
*/

TYPED_TEST(lstm_dynamic_single_layer_test, b16_seq20_is4_hs8_dirs_optional_outputs)
{
    auto batch = 16, max_seq_len = 20, input_size = 4, hidden_size = 8;
    std::vector<float> dynamic_lengths =
    {
        5, 10, 12, 11, 5, 6, 7, 8, 9, 15, 0, 0, 0, 0, 14, 18
    };
    this->single_layer_generic_test(1, batch, max_seq_len, input_size, hidden_size, dynamic_lengths, false, false, false, true, true, 1e-3f);
}

/*
----------------------------------------------
                NEGATIVE TESTS
----------------------------------------------
*/

TEST(lstm_dynamic_negative, wrong_weights_size) {

    auto batch_size = 1, max_sequence_len = 10, input_size = 16, hidden_size = 32, direction = 1;
    auto wrong_value = 50;
    auto& engine = get_test_engine();
    cldnn::data_types dt = cldnn::data_types::f32;
    auto input_mem = engine.allocate_memory({ dt, format::bfyx, { batch_size, max_sequence_len, input_size, 1 } });
    auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, wrong_value } });
    auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, hidden_size, 4 * hidden_size } });
    auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, batch_size, 1 } });
    auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("recurrent", recurrent_mem));
    topology.add(lstm_dynamic("dynamic_lstm",
        "input",
        "dyn_len",
        "weights",
        "recurrent"));
    ASSERT_ANY_THROW(network network(engine, topology));
}

TEST(lstm_dynamic_negative, wrong_recurrent_size_0) {

    auto batch_size = 1, max_sequence_len = 10, input_size = 16, hidden_size = 32, direction = 1;
    auto wrong_value = 50;
    auto& engine = get_test_engine();
    cldnn::data_types dt = cldnn::data_types::f32;
    auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, 1 } });
    auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
    auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, wrong_value, 4 * hidden_size } });
    auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, batch_size, 1 } });
    auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("recurrent", recurrent_mem));
    topology.add(lstm_dynamic("dynamic_lstm",
        "input",
        "dyn_len",
        "weights",
        "recurrent"));
    ASSERT_ANY_THROW(network network(engine, topology));
}

TEST(lstm_dynamic_negative, wrong_recurrent_size_1) {

    auto batch_size = 1, max_sequence_len = 10, input_size = 16, hidden_size = 32, direction = 1;
    auto wrong_value = 50;
    auto& engine = get_test_engine();
    cldnn::data_types dt = cldnn::data_types::f32;
    auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, 1 } });
    auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
    auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, wrong_value, 4 * hidden_size } });
    auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, batch_size, 1 } });
    auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("recurrent", recurrent_mem));
    topology.add(lstm_dynamic("dynamic_lstm",
        "input",
        "dyn_len",
        "weights",
        "recurrent"));
    ASSERT_ANY_THROW(network network(engine, topology));
}

TEST(lstm_dynamic_negative, wrong_dynamic_length_size_0) {

    auto batch_size = 1, max_sequence_len = 10, input_size = 16, hidden_size = 32, direction = 1;
    auto wrong_value = 50;
    auto& engine = get_test_engine();
    cldnn::data_types dt = cldnn::data_types::f32;
    auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, 1 } });
    auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
    auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, hidden_size, 4 * hidden_size } });
    auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, wrong_value, 1 } });
    auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("recurrent", recurrent_mem));
    topology.add(lstm_dynamic("dynamic_lstm",
        "input",
        "dyn_len",
        "weights",
        "recurrent"));
    ASSERT_ANY_THROW(network network(engine, topology));
}

TEST(lstm_dynamic_negative, wrong_dynamic_length_size_1) {

    auto batch_size = 50, max_sequence_len = 10, input_size = 16, hidden_size = 32, direction = 1;
    auto wrong_value = 2;
    auto& engine = get_test_engine();
    cldnn::data_types dt = cldnn::data_types::f32;
    auto input_mem = engine.allocate_memory({ dt, format::bfyx,{ batch_size, max_sequence_len, input_size, 1 } });
    auto weights_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, input_size, 4 * hidden_size } });
    auto recurrent_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, direction, hidden_size, 4 * hidden_size } });
    auto dynamic_length_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, wrong_value, 1 } });
    auto bias_mem = engine.allocate_memory({ dt, format::bfyx,{ 1, 1, 4 * hidden_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(input_layout("dyn_len", dynamic_length_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("recurrent", recurrent_mem));
    topology.add(lstm_dynamic("dynamic_lstm",
        "input",
        "dyn_len",
        "weights",
        "recurrent"));
    ASSERT_ANY_THROW(network network(engine, topology));
}
