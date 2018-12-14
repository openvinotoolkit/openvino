/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/lstm.hpp"
#include <api/CPP/split.hpp>
#include <api/CPP/crop.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/tensor.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>
#include "instrumentation.h"

#include <sstream>
#include <iomanip>


using namespace cldnn;
using namespace tests;

#define FERROR 1E-4

namespace {
    float sigmoid(float x) {
        return 1.f / (1.f + (float)std::exp((float)(-x)));
    }
}

struct offset_order {
    size_t it, ot, ft, zt;
    offset_order(size_t scale, const cldnn_lstm_offset_order& t = cldnn_lstm_offset_order_iofz) {
        static const std::map<cldnn_lstm_offset_order, std::vector<size_t>> offset_map{
            { cldnn_lstm_offset_order_iofz,{ 0, 1, 2, 3 } },
            { cldnn_lstm_offset_order_ifoz,{ 0, 2, 1, 3 } }
        };
        std::vector<size_t> v = offset_map.at(t);
        it = v[0] * scale;
        ot = v[1] * scale;
        ft = v[2] * scale;
        zt = v[3] * scale;
    }
};
cldnn_lstm_offset_order default_offset_type = cldnn_lstm_offset_order_iofz;

template<typename T>
T clip(T val, T threshold) {
    if (threshold > 0) {
        if (val > threshold) return threshold;
        if (val < -threshold) return -threshold;
    }
    return val;
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
                    res += (T)recurrent[0][dir][y][x] * (T)hidden[b][dir][0][x];
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
                     bool hasCell = true, float clip_threshold = 0, bool input_forget = false, size_t dir = 0) {
    size_t hidden_size = tempGEMM[0][0][0].size() / 4;
    size_t batch_size = tempGEMM.size();
    VVVVF<T> tempOut(batch_size, VVVF<T>(2, VVF<T>(1, VF<T>(hidden_size))));
    offset_order off(hidden_size, default_offset_type);

    for (size_t b = 0; b < batch_size; ++b) {
        T *it = &tempGEMM[b][0][0][off.it];
        T *ot = &tempGEMM[b][0][0][off.ot];
        T *ft = &tempGEMM[b][0][0][off.ft];
        T *zt = &tempGEMM[b][0][0][off.zt];
        for (size_t h = 0; h < hidden_size; ++h) {
            T val = sigmoid(clip(it[h], clip_threshold)) * std::tanh((float)clip(zt[h], clip_threshold));
            if (input_forget) {
                val *= (1 - ft[h]);
            }
            if (hasCell) {
                val += cell[b][dir][0][h] * sigmoid(clip(ft[h], clip_threshold));
            }
            tempOut[b][0][0][h] = std::tanh((float)val) * sigmoid(ot[h]);
            tempOut[b][1][0][h] = val;
        }
    }
    return tempOut;
}

template<typename T>
void print(const std::string& s, VVVVF<T>& input) {
    printf("%s -------------\n", s.c_str());
    printf("Size = [%d, %d, %d, %d]\n", (int)input.size(), (int)input[0].size(), (int)input[0][0].size(), (int)input[0][0][0].size());
    for (size_t b = 0; b < input.size(); ++b) {
        for (size_t f = 0; f < input[0].size(); ++f) {
            for (size_t y = 0; y < input[0][0].size(); ++y) {
                for (size_t x = 0; x < input[0][0][0].size(); ++x) {
                    printf("%f ", input[b][f][y][x]);
                }
                printf("\n");
            }
        }
    }
    printf("---------------------------------------\n");
}

// input     = [    batch,  sequence,       direction,      input_size ]
// weights   = [        1, direction, 4 * hidden_size,      input_size ]
// recurrent = [        1, direction, 4 * hidden_size,     hidden_size ]
// biases    = [        1,         1,       direction, 4 * hidden_size ] optional
// cell      = [    batch, direction,               1,     hidden_size ] optional
// hidden    = [    batch, direction,               1,     hidden_size ] optional
// tempGEMM  = [    batch,         1,               1, 4 * hidden_size ] temporary output
// output    = [    batch,  sequence,       direction,     hidden_size ] output
template <typename T>
void lstm_reference(VVVVF<T>& input, VVVVF<T>& hidden, VVVVF<T>& cell, VVVVF<T>& weights, VVVVF<T>& recurrent, VVVVF<T>& bias,
    VVVVF<T>& output, VVVVF<T>& last_hidden, VVVVF<T>& last_cell,
    bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true,
    float clip_threshold = 0, bool input_forget = false, bool scramble_input = true) {
    size_t sequence_len = input[0].size();
    size_t dir_len = weights[0].size();
    size_t batch = input.size();
    size_t input_directions = input[0][0].size();
    for (size_t dir = 0; dir < dir_len; ++dir) {
        bool tempHasInitialHidden = hasInitialHidden;
        bool tempHasInitialCell = hasInitialCell;
        for (size_t seq = 0; seq < sequence_len; ++seq) {
            size_t seq_id = seq;
            size_t input_direction = dir;
            if (scramble_input) {
                if (dir > 0) {
                    seq_id = input_directions == 1 ? sequence_len - seq - 1 : seq;
                    input_direction = input_directions - 1;
                }
            }
            VVVVF<T> tempGEMM = lstm_gemm_reference(input, weights, recurrent, bias, hidden, seq_id, hasBias, tempHasInitialHidden, dir, input_direction);
            VVVVF<T> tempOutput = lstm_elt_reference(tempGEMM, cell, tempHasInitialCell, clip_threshold, input_forget, dir);
            // tempOutput[batch][0] = hidden and tempOutput[batch][1] = cell
            for (size_t i = 0; i < batch; i++) {
                output[i][seq][dir] = tempOutput[i][0][0];
                hidden[i][dir] = tempOutput[i][0];
                cell[i][dir] = tempOutput[i][1];
            }
            tempHasInitialHidden = true;
            tempHasInitialCell = true;
        }
    }
    last_hidden = hidden;
    last_cell = cell;
}



template<typename T>
void generic_lstm_gemm_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
    bool hasBias = true, bool hasHidden = true) {
    int min_random = -2, max_random = 2;

    VVVVF<T> ref_input = generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);

    VVVVF<T> ref_output = lstm_gemm_reference(ref_input, ref_weights, ref_recurrent, ref_bias, ref_hidden, 0, hasBias, hasHidden);

    engine engine;
    memory input = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size,   sequence_len,  input_size,      1 } });
    memory weights = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,            direction,     input_size,      4 * hidden_size } });
    memory recurrent = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,            direction,     hidden_size,     4 * hidden_size } });
    memory biases = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,            1,             4 * hidden_size, direction } });
    memory hidden = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size,   direction,     hidden_size,     1 } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    if (hasBias) {
        topology.add(data("biases", biases));
    }
    if (hasHidden) {
        topology.add(input_layout("hidden", hidden.get_layout()));
    }

    topology.add(lstm_gemm("lstm_gemm", "input", "weights", "recurrent", hasBias ? "biases" : "", hasHidden ? "hidden" : ""));

    network network(engine, topology);
    network.set_input_data("input", input);
    if (hasHidden) {
        network.set_input_data("hidden", hidden);
    }

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int x = 0; x < 4 * hidden_size; ++x)
            EXPECT_EQ(ref_output[b][0][0][x], output_ptr[i++]);
    }
}

template<typename T>
void generic_lstm_elt_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size, bool hasCell = true,
    float clip_threshold = 0.f, bool input_forget = false) {
    // tempGEMM  = [        1, direction,           batch, 4 * hidden_size ] input
    // cell      = [        1, direction,           batch,     hidden_size ] optional
    // output    = [        2, direction,           batch,     hidden_size ] output concat[hidden, cell]
    int min_random = -2, max_random = 2;

    VVVVF<T> ref_tempGEMM = generate_random_4d<T>(batch_size, direction, 1, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VF<T> ref_tempGEMM_vec = flatten_4d<T>(cldnn::format::bfyx, ref_tempGEMM);
    VF<T> ref_cell_vec = flatten_4d<T>(cldnn::format::bfyx, ref_cell);

    VVVVF<T> ref_output = lstm_elt_reference(ref_tempGEMM, ref_cell, hasCell, clip_threshold, input_forget);

    engine engine;
    memory tempGEMM = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size,    direction, 4 * hidden_size, 1 } });
    memory cell = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size,    direction,     hidden_size, 1 } });
    set_values(tempGEMM, ref_tempGEMM_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    topology.add(input_layout("tempGEMM", tempGEMM.get_layout()));
    if (hasCell) {
        topology.add(input_layout("cell", cell.get_layout()));
    }
    topology.add(lstm_elt("lstm_elt", "tempGEMM", hasCell ? "cell" : "", clip_threshold, input_forget));

    network network(engine, topology);
    network.set_input_data("tempGEMM", tempGEMM);
    if (hasCell) {
        network.set_input_data("cell", cell);
    }

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < 2; ++j) {
            for (int x = 0; x < hidden_size; ++x)
            {
                auto idx = b * 2 * hidden_size + j * hidden_size + x;
                EXPECT_NEAR(ref_output[b][j][0][x], output_ptr[idx], FERROR);
            }
        }
    }
}

std::string get_string_id(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

// --------------- Manually constructed LSTM ----------------------------------------
// This function manually generates an lstm node sequence by conbining lstm_gemm and lstm_elt nodes
// it requires that the output of the lstm_elt node is croped to obtain the corresponding hidden and cell outputs
void generate_lstm_topology(topology& t, memory& input, memory& hidden, memory& cell,
    memory& weights, memory& recurrent, memory& biases, int sequence_len,
    bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    auto hidden_size = hidden.get_layout().size;
    t.add(input_layout("input", input.get_layout()));
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<primitive_id> output_ids_offsets;
    for (int i = 0; i < sequence_len; ++i)
        input_ids_offsets.push_back({ get_string_id(i),{ 0, i, 0, 0 } });
    t.add(split("inputSplit", "input", input_ids_offsets));
    t.add(data("weights", weights));
    t.add(data("recurrent", recurrent));

    std::string biasStr = "";
    std::string hiddenStr = "";
    std::string cellStr = "";
    if (hasBias)
    {
        t.add(data("biases", biases));
        biasStr = "biases";
    }
    if (hasInitialHidden)
    {
        t.add(input_layout("hidden", hidden.get_layout()));
        hiddenStr = "hidden";
    }
    if (hasInitialCell)
    {
        t.add(input_layout("cell", cell.get_layout()));
        cellStr = "cell";
    }
    for (int i = 0; i < sequence_len; ++i) {
        std::string lstm_gemm_id = "lstm_gemm" + get_string_id(i);
        std::string lstm_elt_id = "lstm_elt" + get_string_id(i);
        std::string crop_id = "crop" + get_string_id(i);

        t.add(lstm_gemm(lstm_gemm_id, "inputSplit:" + get_string_id(i), "weights", "recurrent", biasStr, hiddenStr));
        t.add(lstm_elt(lstm_elt_id, lstm_gemm_id, cellStr));

        hiddenStr = crop_id + ":hidden";
        t.add(crop(hiddenStr, lstm_elt_id, hidden_size, tensor{ 0,0,0,0 }));
        if (i < sequence_len - 1) {
            cellStr = crop_id + ":cell";
            t.add(crop(cellStr, lstm_elt_id, hidden_size, tensor{ 0,1,0,0 }));
        }
        output_ids_offsets.push_back(hiddenStr);
    }
    t.add(concatenation("concatenation", output_ids_offsets, concatenation::along_f));
}


template<typename T>
void generic_lstm_custom_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
    bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    std::cout << "Input Size = " << input_size << " Hidden Size = " << hidden_size << " Sequence Len = " << sequence_len << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;
    VVVVF<T> ref_input = generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VVVVF<T> ref_output(batch_size, VVVF<T>(sequence_len, VVF<T>(direction, VF<T>(hidden_size))));
    VVVVF<T> last_hidden(batch_size, VVVF<T>(direction, VVF<T>(1, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(direction, VVF<T>(1, VF<T>(hidden_size))));

    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);
    VF<T> ref_cell_vec = flatten_4d<T>(cldnn::format::bfyx, ref_cell);
    lstm_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output, last_hidden, last_cell,
        hasBias, hasInitialHidden, hasInitialCell);

    engine engine;
    memory input = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size, sequence_len,  input_size,       1 } });
    memory weights = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,          direction,     input_size,       4 * hidden_size } });
    memory recurrent = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,          direction,     hidden_size,      4 * hidden_size } });
    memory biases = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ 1,          1,             4 * hidden_size,  direction } });
    memory hidden = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size, direction,     hidden_size,      1 } });
    memory cell = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx,{ batch_size, direction,     hidden_size,      1 } });
    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    generate_lstm_topology(topology, input, hidden, cell, weights, recurrent, biases, sequence_len,
        hasBias, hasInitialHidden, hasInitialCell);

    network network(engine, topology);
    network.set_input_data("input", input);
    if (hasInitialHidden) network.set_input_data("hidden", hidden);
    if (hasInitialCell) network.set_input_data("cell", cell);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    size_t output_size = outputs.begin()->second.get_memory().size() / sizeof(T);
    ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

    auto output = outputs.begin()->second.get_memory();
    auto output_ptr = output.pointer<T>();
    int i = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < sequence_len; ++s) {
            for (int x = 0; x < hidden_size; ++x) {
                for (int d = 0; d < direction; ++d) {
                    ASSERT_NEAR(ref_output[b][s][d][x], output_ptr[i++], FERROR);
                }
            }
        }
    }
}

// -------------------------------------------------------
template<typename T>
void generic_lstm_gpu_test(int layers, int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
                            bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true,
                            float clip_threshold = 0, bool input_forget = false) {
    std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
            << " Sequence Len = " << sequence_len << " Direction = " << direction << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;

    VVVVF<T> ref_input = generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);

    std::vector<VVVVF<T>> ref_weights;
    std::vector<VVVVF<T>> ref_recurrent;
    std::vector<VVVVF<T>> ref_bias;
    std::vector<VVVVF<T>> ref_hidden;
    std::vector<VVVVF<T>> ref_cell;
    std::vector<VVVVF<T>> ref_output;

    for (int i = 0; i < layers; ++i) {
        ref_weights.push_back(generate_random_4d<T>(1, direction, 4 * hidden_size, i==0 ? input_size : hidden_size, min_random, max_random));
        ref_recurrent.push_back(generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random));
        ref_bias.push_back(generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random));
        ref_hidden.push_back(generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random));
        ref_cell.push_back(generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random));
        ref_output.push_back(VVVVF<T>(batch_size, VVVF<T>(sequence_len, VVF<T>(direction, VF<T>(hidden_size)))));
    }

    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    std::vector<VF<T>> ref_weights_vec;
    std::vector<VF<T>> ref_recurrent_vec;
    std::vector<VF<T>> ref_bias_vec;
    std::vector<VF<T>> ref_hidden_vec;
    std::vector<VF<T>> ref_cell_vec;
    for (int i = 0; i < layers; ++i) {
        ref_weights_vec.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_weights[i]));
        ref_recurrent_vec.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_recurrent[i]));
        ref_bias_vec.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_bias[i]));
        ref_hidden_vec.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_hidden[i]));
        ref_cell_vec.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_cell[i]));
    }

    VVVVF<T> last_hidden(batch_size, VVVF<T>(direction, VVF<T>(1, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(direction, VVF<T>(1, VF<T>(hidden_size))));

    lstm_reference(ref_input, ref_hidden[0], ref_cell[0], ref_weights[0], ref_recurrent[0], ref_bias[0], ref_output[0],
                   last_hidden, last_cell, hasBias, hasInitialHidden, hasInitialCell,
                   clip_threshold, input_forget, true);

    for (int i = 1; i < layers; ++i) {
        lstm_reference(ref_output[i - 1], ref_hidden[i], ref_cell[i], ref_weights[i], ref_recurrent[i],
                        ref_bias[i], ref_output[i],
                        last_hidden, last_cell, hasBias, hasInitialHidden, hasInitialCell,
                        clip_threshold, input_forget, false);
    }

    engine engine;

    memory input = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, {batch_size, sequence_len, input_size, 1} });
    set_values(input, ref_input_vec);

    std::vector<memory> weights;
    std::vector<memory> recurrent;
    std::vector<memory> biases;
    std::vector<memory> hidden;
    std::vector<memory> cell;
    for(int i = 0; i < layers; ++i) {
        weights.push_back(memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, direction, i==0 ? input_size : hidden_size, 4 * hidden_size } }));
        set_values(weights[i], ref_weights_vec[i]);
        recurrent.push_back(memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, direction, hidden_size, 4 * hidden_size } }));
        set_values(recurrent[i], ref_recurrent_vec[i]);
        if (hasBias) {
            biases.push_back(memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1, 1, 4 * hidden_size, direction } }));
            set_values(biases[i], ref_bias_vec[i]);
        }
        if (hasInitialHidden) {
            hidden.push_back(memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { batch_size,  direction, hidden_size, 1 } }));
            set_values(hidden[i], ref_hidden_vec[i]);
        }
        if (hasInitialCell) {
            cell.push_back(memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { batch_size, direction, hidden_size, 1 } }));
            set_values(cell[i], ref_cell_vec[i]);
        }
    }

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<primitive_id> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input.get_layout()));
    for (int i = 0; i < sequence_len; ++i) {
        input_ids_offsets.push_back({get_string_id(i), {0, i, 0, 0}});
        lstm_inputs.push_back("inputSplit:"+get_string_id(i));
    }
    topology.add(split("inputSplit", "input", input_ids_offsets));
    cldnn::primitive_id prev_lstm_id;
    for(int i = 0; i < layers; ++i) {
        std::string sid = get_string_id(i);
        std::string lstm_id = "lstm" + sid;
        std::string weights_id = "weights" + sid;
        std::string recurrent_id = "recurrent" + sid;
        std::string biases_id = "biases" + sid;
        std::string hidden_id = "hidden" + sid;
        std::string cell_id = "cell" + sid;

        topology.add(data(weights_id, weights[i]));
        topology.add(data(recurrent_id, recurrent[i]));
        if (hasBias) topology.add(data(biases_id, biases[i]));
        if (hasInitialHidden) topology.add(input_layout(hidden_id, hidden[i].get_layout()));
        if (hasInitialCell) topology.add(input_layout(cell_id, cell[i].get_layout()));
        if (i == 0) {
            topology.add(lstm(lstm_id, lstm_inputs, weights_id, recurrent_id,
                            hasBias ? biases_id : "", hasInitialHidden ? hidden_id : "", hasInitialCell ? cell_id : "", "",
                            clip_threshold, input_forget, {}, {}, default_offset_type));
        }
        else {
            topology.add(lstm(lstm_id, { prev_lstm_id }, weights_id, recurrent_id,
                            hasBias ? biases_id : "", hasInitialHidden ? hidden_id : "", hasInitialCell ? cell_id : "", "",
                            clip_threshold, input_forget, {}, {}, default_offset_type));
        }
        prev_lstm_id = lstm_id;
    }

    network network(engine, topology);
    network.set_input_data("input", input);
    for (int i = 0; i < layers; ++i) {
        std::string sid = get_string_id(i);
        if (hasInitialHidden) network.set_input_data("hidden" + sid, hidden[i]);
        if (hasInitialCell) network.set_input_data("cell" + sid, cell[i]);
    }
    auto outputs = network.execute();
    {
        ASSERT_EQ(outputs.size(), size_t(1));
        size_t output_size = outputs.begin()->second.get_memory().size() / sizeof(T);
        ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

        auto output = outputs.begin()->second.get_memory();
        
        // Get the output tensor
        cldnn::layout output_layout = output.get_layout();
        cldnn::tensor output_tensor = output_layout.size; 
        
        // Compare the output tensor configuration against the reference value
        // Output tensor is configured in bfyx format
        ASSERT_EQ(batch_size, output_tensor.batch[0]);
        ASSERT_EQ(sequence_len, output_tensor.feature[0]);
        ASSERT_EQ(direction, output_tensor.spatial[1]);
        ASSERT_EQ(hidden_size, output_tensor.spatial[0]); 

        auto output_ptr = output.pointer<T>();
        int32_t i = 0;
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t s = 0; s < sequence_len; ++s) {
                for (int32_t d = 0; d < direction; ++d) {
                    for (int32_t x = 0; x <  hidden_size; ++x) {
                        ASSERT_NEAR(ref_output[layers-1][b][s][d][x], output_ptr[i++], FERROR);
                    }
                }
            }
        }
    }
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_test_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, false);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_clip_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.3f);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_input_forget_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.f, 1);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_clip_input_forget_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.5f, 1);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_biasf32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, false);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_cell_f32) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, false);
}

// generic_lstm_gpu_test paramters:
// layers, sequence, dir, batch, input, hidden, bias, initial_h, initial_cell, threshold, coupled_input_forget
TEST(lstm_gpu, generic_lstm_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, false, true, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, true);
}

TEST(lstm_gpu, generic_lstm_no_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, true, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, true, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, false);
}

TEST(lstm_gpu, generic_lstm_clip_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 0);
}

TEST(lstm_gpu, generic_lstm_input_forget_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.f, 1);
}

TEST(lstm_gpu, generic_lstm_clip_input_forget_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 1);
}

TEST(lstm_gpu, generic_lstm_offset_order_ifoz_f32) {
    default_offset_type = cldnn_lstm_offset_order_ifoz;
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true);
    default_offset_type = cldnn_lstm_offset_order_iofz;
}

TEST(lstm_gpu, generic_lstm_canonical_f32) {
    generic_lstm_gpu_test<float>(1, 1, 1, 1, 1, 1, true, true, true);
}

// bidirectional support
TEST(lstm_gpu, generic_lstm_bi_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, false, false, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, false, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, true);
}

// multi-layer support
TEST(lstm_gpu, generic_lstm_stacked_no_seq_f32) {
    generic_lstm_gpu_test<float>(4, 1, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_f32) {
    generic_lstm_gpu_test<float>(4, 7, 1, 3, 3, 2, true, true, true);
}

TEST(lstm_gpu, generic_lstm_stacked_bi_f32) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_bi_f32) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true);
}

// TODO: Add tests for the following:
// optional concatenate output
// optional last hidden
// optional last cell
// optional activation list

