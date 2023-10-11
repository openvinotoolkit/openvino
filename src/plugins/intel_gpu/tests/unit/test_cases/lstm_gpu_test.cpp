// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/lstm.hpp>
#include <intel_gpu/primitives/split.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <sstream>
#include <iomanip>

#ifdef _MSC_VER
# pragma warning(disable: 4503)
#endif

using namespace cldnn;
using namespace ::tests;

#define FERROR 1E-4

namespace {
float sigmoid(float x) {
    return 1.f / (1.f + (float)std::exp((float)(-x)));
}
struct offset_order {
    size_t it, ot, ft, zt;
    offset_order(size_t scale, const lstm_weights_order& t = lstm_weights_order::iofz) {
        static const std::map<lstm_weights_order, std::vector<size_t>> offset_map{
            { lstm_weights_order::iofz,{ 0, 1, 2, 3 } },
            { lstm_weights_order::ifoz,{ 0, 2, 1, 3 } }
        };
        std::vector<size_t> v = offset_map.at(t);
        it = v[0] * scale;
        ot = v[1] * scale;
        ft = v[2] * scale;
        zt = v[3] * scale;
    }
};
lstm_weights_order default_offset_type = lstm_weights_order::iofz;
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
                    res += (T)recurrent[0][dir][y][x] * (T)hidden[b][0][dir][x];
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
    offset_order off(hidden_size, default_offset_type);

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
            // data type may be float or ov::float16 (half)
            tempOut[b][0][0][h] = (T)(std::tanh(val) * sigmoid(fp32_ot));
            tempOut[b][1][0][h] = (T)val;
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
void lstm_reference(VVVVF<T>& input, VVVVF<T>& hidden, VVVVF<T>& cell,
                    VVVVF<T>& weights, VVVVF<T>& recurrent, VVVVF<T>& bias,
                    VVVVF<T>& output, VVVVF<T>& last_hidden,
                    VVVVF<T>& last_cell, bool hasBias = true,
                    bool hasInitialHidden = true, bool hasInitialCell = true,
                    float clip_threshold = 0, bool input_forget = false,
                    bool scramble_input = true)
{
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
                hidden[i][0][dir] = tempOutput[i][0][0];
                cell[i][0][dir] = tempOutput[i][1][0];
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
    bool hasBias, bool hasHidden, bool is_caching_test = false) {
    int min_random = -2, max_random = 2;

    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = rg.generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = rg.generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = rg.generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = rg.generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);

    VVVVF<T> ref_output = lstm_gemm_reference(ref_input, ref_weights, ref_recurrent, ref_bias, ref_hidden, 0, hasBias, hasHidden);

    constexpr auto dt = std::is_same<T, float>::value ? data_types::f32 : data_types::f16;
    auto& engine = get_test_engine();

    // If the input is of fp16 type then, the memory::ptr will be allocated as such
    if (!engine.get_device_info().supports_fp16)
    {
        if (dt == data_types::f16)
        {
            return;
        }
    }

    memory::ptr input = engine.allocate_memory({ dt, format::bfyx,     { batch_size,   sequence_len,  input_size,      1 } });
    memory::ptr weights = engine.allocate_memory({ dt, format::bfyx,   { 1,            direction,     input_size,      4 * hidden_size } });
    memory::ptr recurrent = engine.allocate_memory({ dt, format::bfyx, { 1,            direction,     hidden_size,     4 * hidden_size } });
    memory::ptr biases = engine.allocate_memory({ dt, format::bfyx,    { 1,            1,             4 * hidden_size, direction } });
    memory::ptr hidden = engine.allocate_memory({ dt, format::bfyx,    { batch_size,   direction,     hidden_size,     1 } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    if (hasBias) {
        topology.add(data("biases", biases));
    }
    if (hasHidden) {
        topology.add(input_layout("hidden", hidden->get_layout()));
    }

    topology.add(lstm_gemm("lstm_gemm", input_info("input"), "weights", "recurrent", hasBias ? "biases" : "", hasHidden ? "hidden" : ""));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    if (hasHidden) {
        network->set_input_data("hidden", hidden);
    }

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());
    int i = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int x = 0; x < 4 * hidden_size; ++x)
            ASSERT_FLOAT_EQ(ref_output[b][0][0][x], output_ptr[i++]);
    }
}

template<typename T>
void generic_lstm_elt_gpu_test(int /* sequence_len */, int direction, int batch_size,
    int /* input_size */, int hidden_size, bool hasCell,
    T clip_threshold, bool input_forget, bool is_caching_test = false) {
    // tempGEMM  = [        1, direction,           batch, 4 * hidden_size ] input
    // cell      = [        1, direction,           batch,     hidden_size ] optional
    // output    = [        2, direction,           batch,     hidden_size ] output concat[hidden, cell]
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> ref_tempGEMM = rg.generate_random_4d<T>(batch_size, direction, 1, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = rg.generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VF<T> ref_tempGEMM_vec = flatten_4d<T>(cldnn::format::bfyx, ref_tempGEMM);
    VF<T> ref_cell_vec = flatten_4d<T>(cldnn::format::bfyx, ref_cell);

    VVVVF<T> ref_output = lstm_elt_reference(ref_tempGEMM, ref_cell, hasCell, clip_threshold, input_forget);

    // We observe some mismatch in down-converting from fp32 to fp16
    // between the reference implementation and opencl kernel. This can be
    // a simple rounding error. Thus, for fp16 we are increasing our tolerance
    // to error from 1E-4 to 1E-2
    constexpr float ferror = std::is_same<T, float>::value ? (float)1E-4 : (float)1E-2;
    constexpr auto dt = std::is_same<T, float>::value ? data_types::f32 : data_types::f16;
    auto& engine = get_test_engine();

    // If the input is of fp16 type then, the memory::ptr will be allocated as such
    if (!engine.get_device_info().supports_fp16)
    {
        if (dt == data_types::f16)
        {
            return;
        }
    }

    memory::ptr tempGEMM = engine.allocate_memory({ dt, format::bfyx,{ batch_size,    direction, 4 * hidden_size, 1 } });
    memory::ptr cell = engine.allocate_memory({ dt, format::bfyx,{ batch_size,    direction,     hidden_size, 1 } });
    set_values(tempGEMM, ref_tempGEMM_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    topology.add(input_layout("tempGEMM", tempGEMM->get_layout()));
    if (hasCell) {
        topology.add(input_layout("cell", cell->get_layout()));
    }
    topology.add(lstm_elt("lstm_elt", input_info("tempGEMM"), hasCell ? "cell" : "", clip_threshold, input_forget));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("tempGEMM", tempGEMM);
    if (hasCell) {
        network->set_input_data("cell", cell);
    }

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < 2; ++j) {
            for (int x = 0; x < hidden_size; ++x)
            {
                auto idx = b * 2 * hidden_size + j * hidden_size + x;
                ASSERT_NEAR(ref_output[b][j][0][x], output_ptr[idx] , ferror);
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
void generate_lstm_topology(topology& t, memory::ptr input, memory::ptr hidden, memory::ptr cell,
    memory::ptr weights, memory::ptr recurrent, memory::ptr biases, int sequence_len,
    bool hasBias = true, bool hasInitialHidden = true, bool hasInitialCell = true) {
    auto hidden_size = hidden->get_layout().get_tensor();
    t.add(input_layout("input", input->get_layout()));
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<input_info> output_ids_offsets;
    for (int i = 0; i < sequence_len; ++i)
        input_ids_offsets.push_back({ get_string_id(i),{ 0, i, 0, 0 } });
    t.add(split("inputSplit", input_info("input"), input_ids_offsets));
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
        t.add(input_layout("hidden", hidden->get_layout()));
        hiddenStr = "hidden";
    }
    if (hasInitialCell)
    {
        t.add(input_layout("cell", cell->get_layout()));
        cellStr = "cell";
    }
    for (int i = 0; i < sequence_len; ++i) {
        std::string lstm_gemm_id = "lstm_gemm" + get_string_id(i);
        std::string lstm_elt_id = "lstm_elt" + get_string_id(i);
        std::string crop_id = "crop" + get_string_id(i);

        t.add(lstm_gemm(lstm_gemm_id, input_info("inputSplit:" + get_string_id(i)), "weights", "recurrent", biasStr, hiddenStr));
        t.add(lstm_elt(lstm_elt_id, input_info(lstm_gemm_id), cellStr));

        hiddenStr = crop_id + ":hidden";
        t.add(crop(hiddenStr, input_info(lstm_elt_id), hidden_size, tensor{ 0,0,0,0 }));
        if (i < sequence_len - 1) {
            cellStr = crop_id + ":cell";
            t.add(crop(cellStr, input_info(lstm_elt_id), hidden_size, tensor{ 0,1,0,0 }));
        }
        output_ids_offsets.push_back(input_info(hiddenStr));
    }
    t.add(concatenation("concatenation", output_ids_offsets, 1));
}

template<typename T>
void generic_lstm_custom_gpu_test(int sequence_len, int direction, int batch_size, int input_size, int hidden_size,
    bool hasBias, bool hasInitialHidden, bool hasInitialCell, bool is_caching_test = false) {
    std::cout << "Input Size = " << input_size << " Hidden Size = " << hidden_size << " Sequence Len = " << sequence_len << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);
    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = rg.generate_random_4d<T>(1, direction, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = rg.generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = rg.generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = rg.generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = rg.generate_random_4d<T>(batch_size, direction, 1, hidden_size, min_random, max_random);
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

    auto& engine = get_test_engine();
    memory::ptr input = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ batch_size, sequence_len,  input_size,       1 } });
    memory::ptr weights = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ 1,          direction,     input_size,       4 * hidden_size } });
    memory::ptr recurrent = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ 1,          direction,     hidden_size,      4 * hidden_size } });
    memory::ptr biases = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ 1,          1,             4 * hidden_size,  direction } });
    memory::ptr hidden = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ batch_size, direction,     hidden_size,      1 } });
    memory::ptr cell = engine.allocate_memory({ ov::element::from<T>(), format::bfyx,{ batch_size, direction,     hidden_size,      1 } });
    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    generate_lstm_topology(topology, input, hidden, cell, weights, recurrent, biases, sequence_len,
        hasBias, hasInitialHidden, hasInitialCell);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    if (hasInitialHidden) network->set_input_data("hidden", hidden);
    if (hasInitialCell) network->set_input_data("cell", cell);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    size_t output_size = outputs.begin()->second.get_memory()->size() / sizeof(T);
    ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());
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
                            bool hasBias, bool hasInitialHidden, bool hasInitialCell,
                            T clip_threshold, bool input_forget, bool is_caching_test = false) {
    std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
            << " Sequence Len = " << sequence_len << " Direction = " << direction << " Batch Size = " << batch_size << std::endl;
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);

    std::vector<VVVVF<T>> ref_weights;
    std::vector<VVVVF<T>> ref_recurrent;
    std::vector<VVVVF<T>> ref_bias;
    std::vector<VVVVF<T>> ref_hidden;
    std::vector<VVVVF<T>> ref_cell;
    std::vector<VVVVF<T>> ref_output;

    for (int i = 0; i < layers; ++i) {
        ref_weights.push_back(rg.generate_random_4d<T>(1, direction, 4 * hidden_size, i==0 ? input_size : hidden_size, min_random, max_random));
        ref_recurrent.push_back(rg.generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random));
        ref_bias.push_back(rg.generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random));
        ref_hidden.push_back(rg.generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random));
        ref_cell.push_back(rg.generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random));
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

    VVVVF<T> last_hidden(batch_size, VVVF<T>(1, VVF<T>(direction, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(1, VVF<T>(direction, VF<T>(hidden_size))));

    lstm_reference(ref_input, ref_hidden[0], ref_cell[0], ref_weights[0], ref_recurrent[0], ref_bias[0], ref_output[0],
                   last_hidden, last_cell, hasBias, hasInitialHidden, hasInitialCell,
                   clip_threshold, input_forget, true);

    for (int i = 1; i < layers; ++i) {
        lstm_reference(ref_output[i - 1], ref_hidden[i], ref_cell[i], ref_weights[i], ref_recurrent[i],
                        ref_bias[i], ref_output[i],
                        last_hidden, last_cell, hasBias, hasInitialHidden, hasInitialCell,
                        clip_threshold, input_forget, false);
    }

    // We observe some mismatch in down-converting from fp32 to fp16
    // between the reference implementation and opencl kernel. This can be
    // a simple rounding error. Thus, for fp16 we are increasing our tolerance
    // to error from 1E-4 to 1E-2
    constexpr float ferror = std::is_same<T, float>::value ? (float)1E-4 : (float)1E-2;
    constexpr auto dt = std::is_same<T, float>::value ? data_types::f32 : data_types::f16;
    auto& engine = get_test_engine();

    // If the input is of fp16 type then, the memory::ptr will be allocated as such
    if (!engine.get_device_info().supports_fp16)
    {
        if (dt == data_types::f16)
        {
            return;
        }
    }

    memory::ptr input = engine.allocate_memory({ dt, format::bfyx, {batch_size, sequence_len, input_size, 1} });
    set_values(input, ref_input_vec);

    std::vector<memory::ptr> weights;
    std::vector<memory::ptr> recurrent;
    std::vector<memory::ptr> biases;
    std::vector<memory::ptr> hidden;
    std::vector<memory::ptr> cell;
    for(int i = 0; i < layers; ++i) {
        weights.push_back(engine.allocate_memory({ dt, format::bfyx, { 1, direction, i==0 ? input_size : hidden_size, 4 * hidden_size } }));
        set_values(weights[i], ref_weights_vec[i]);
        recurrent.push_back(engine.allocate_memory({ dt, format::bfyx, { 1, direction, hidden_size, 4 * hidden_size } }));
        set_values(recurrent[i], ref_recurrent_vec[i]);
        if (hasBias) {
            biases.push_back(engine.allocate_memory({ dt, format::bfyx, { 1, 1, 4 * hidden_size, direction } }));
            set_values(biases[i], ref_bias_vec[i]);
        }
        if (hasInitialHidden) {
            hidden.push_back(engine.allocate_memory({ dt, format::bfyx, { batch_size, 1, hidden_size, direction } }));
            set_values(hidden[i], ref_hidden_vec[i]);
        }
        if (hasInitialCell) {
            cell.push_back(engine.allocate_memory({ dt, format::bfyx, { batch_size, 1, hidden_size, direction} }));
            set_values(cell[i], ref_cell_vec[i]);
        }
    }

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<input_info> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input->get_layout()));
    for (int i = 0; i < sequence_len; ++i) {
        input_ids_offsets.push_back({get_string_id(i), {0, i, 0, 0}});
        lstm_inputs.push_back(input_info("inputSplit:"+get_string_id(i)));
    }
    topology.add(split("inputSplit", input_info("input"), input_ids_offsets));
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
        if (hasInitialHidden) topology.add(input_layout(hidden_id, hidden[i]->get_layout()));
        if (hasInitialCell) topology.add(input_layout(cell_id, cell[i]->get_layout()));
        if (i == 0) {
            topology.add(lstm(lstm_id, lstm_inputs, weights_id, recurrent_id,
                            hasBias ? biases_id : "", hasInitialHidden ? hidden_id : "", hasInitialCell ? cell_id : "", "",
                            clip_threshold, input_forget,
                            { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                            lstm_output_selection::sequence, default_offset_type));
        }
        else {
            topology.add(lstm(lstm_id, { input_info(prev_lstm_id) }, weights_id, recurrent_id,
                            hasBias ? biases_id : "", hasInitialHidden ? hidden_id : "", hasInitialCell ? cell_id : "", "",
                            clip_threshold, input_forget,
                            { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                            lstm_output_selection::sequence, default_offset_type));
        }
        prev_lstm_id = lstm_id;
    }

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    for (int i = 0; i < layers; ++i) {
        std::string sid = get_string_id(i);
        if (hasInitialHidden) network->set_input_data("hidden" + sid, hidden[i]);
        if (hasInitialCell) network->set_input_data("cell" + sid, cell[i]);
    }
    auto outputs = network->execute();
    {
        ASSERT_EQ(outputs.size(), size_t(1));
        size_t output_size = outputs.begin()->second.get_memory()->size() / sizeof(T);
        ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

        auto output = outputs.begin()->second.get_memory();

        // Get the output tensor
        cldnn::layout output_layout = output->get_layout();

        // Compare the output tensor configuration against the reference value
        // Output tensor is configured in bfyx format
        ASSERT_EQ(batch_size, output_layout.batch());
        ASSERT_EQ(sequence_len, output_layout.feature());
        ASSERT_EQ(direction, output_layout.spatial(1));
        ASSERT_EQ(hidden_size, output_layout.spatial(0));

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());
        int32_t i = 0;
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t s = 0; s < sequence_len; ++s) {
                for (int32_t d = 0; d < direction; ++d) {
                    for (int32_t x = 0; x <  hidden_size; ++x) {
                        ASSERT_NEAR(ref_output[layers - 1][b][s][d][x], output_ptr[i++], ferror);
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------
template<typename T>
void lstm_gpu_output_test(const lstm_output_selection& output_selection, int directions, bool is_caching_test = false) {
    int layers = 1;
    int sequence_len = 4;
    int batch_size = 3;
    int input_size = 3;
    int hidden_size = 4;

    std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
            << " Sequence Len = " << sequence_len << " Directions = " << directions << " Batch Size = " << batch_size
			<< " Output selection: " << static_cast<int>(output_selection) << std::endl;
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = rg.generate_random_4d<T>(1, 1, directions, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_output = VVVVF<T>(batch_size, VVVF<T>(sequence_len, VVF<T>(directions, VF<T>(hidden_size))));

    VF<T> ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(cldnn::format::bfyx, ref_hidden);
    VF<T> ref_cell_vec = flatten_4d<T>(cldnn::format::bfyx, ref_cell);

    VVVVF<T> last_hidden(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));

    lstm_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output,
                   last_hidden, last_cell, true, true, true,
                   (T)0, false, true);

    auto& engine = get_test_engine();

    memory::ptr input = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {batch_size, sequence_len, input_size, 1} });
    memory::ptr weights = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, input_size , 4 * hidden_size } });
    memory::ptr recurrent = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, hidden_size, 4 * hidden_size } });
    memory::ptr biases = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, 1, 4 * hidden_size, directions } });
    memory::ptr hidden = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, directions } });
    memory::ptr cell = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, directions } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    bool emit_last_cell = output_selection == lstm_output_selection::hidden_cell ||
                          output_selection == lstm_output_selection::sequence_cell;
    bool emit_last_hidden = output_selection == lstm_output_selection::hidden ||
                            output_selection == lstm_output_selection::hidden_cell;

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<input_info> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input->get_layout()));
    for (int i = 0; i < sequence_len; ++i)
    {
        input_ids_offsets.push_back({get_string_id(i), {0, i, 0, 0}});
        lstm_inputs.push_back(input_info("inputSplit:"+get_string_id(i)));
    }
    topology.add(split("inputSplit", input_info("input"), input_ids_offsets));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    topology.add(data("biases", biases));
    topology.add(input_layout("hidden", hidden->get_layout()));
    topology.add(input_layout("cell", cell->get_layout()));
    topology.add(lstm("lstm", lstm_inputs, "weights", "recurrent",
                      "biases", "hidden", "cell", "", 0, false,
                      { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                      output_selection, default_offset_type));
    if (emit_last_cell)
    {
        int32_t concatenation_len = emit_last_hidden ? 2 : sequence_len + 1;
        tensor hidden_tensor {batch_size, concatenation_len - 1, hidden_size, directions};
        tensor cell_tensor {batch_size, 1, hidden_size, directions};
        topology.add(crop(emit_last_hidden ? "crop:last_hidden" : "crop:sequence", input_info("lstm"), hidden_tensor, tensor{0, 0, 0, 0}));
        topology.add(crop("crop:last_cell", input_info("lstm"), cell_tensor, tensor{0, concatenation_len - 1, 0, 0}));
    }

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    network->set_input_data("hidden", hidden);
    network->set_input_data("cell", cell);

    auto outputs = network->execute();
	uint32_t ref_num_output_primitives = 1;  // Output will return atleast 1 primitive

	if (emit_last_cell) {
		// add another primitve to account for cell state if the output selection includes cell state
		ref_num_output_primitives += 1;
	}

	// check if the number of returned primitives match the expected number of output primitives
	ASSERT_EQ(ref_num_output_primitives, outputs.size());

	for (auto itr = outputs.begin(); itr != outputs.end(); itr++)
	{
        auto output_layout = itr->second.get_memory()->get_layout();
        primitive_id primitive_name = itr->first;

		cldnn::memory::ptr output_memory = itr->second.get_memory();
        int32_t output_size = (int32_t)(itr->second.get_memory()->size() / sizeof(T));
		cldnn::tensor ref_output_tensor;
		VVVVF<T> ref_primitive_output;

		int32_t ref_batch_size = batch_size;
		int32_t ref_hidden_size = hidden_size;
		int32_t ref_directions = directions;

        int32_t ref_seq_len = 1;
        // Set the reference output against which the primitive's output will be compared
		if (primitive_name.find("crop:last_cell") != std::string::npos)
		{
			ref_primitive_output = last_cell;
		}
		else if (emit_last_hidden || primitive_name.find("crop:last_hidden") != std::string::npos)
		{
			ref_primitive_output = last_hidden;
		}
		else
		{
			ref_seq_len = sequence_len;
			ref_primitive_output = ref_output;
		}

		ref_output_tensor = { ref_batch_size, ref_seq_len, ref_hidden_size, ref_directions };
		int32_t ref_output_size = ref_batch_size * ref_seq_len * ref_hidden_size * ref_directions;

		// The number of elements in reference should match the number of elements in the primitive's output
		ASSERT_EQ(ref_output_size , output_size);

        // Compare the output tensor configuration against the reference value
        // Output tensor is configured in bfyx format
        ASSERT_EQ(ref_batch_size, output_layout.batch());
        ASSERT_EQ(ref_seq_len, output_layout.feature());		// Sequence length should match
		ASSERT_EQ(ref_directions, output_layout.spatial(1));	// directions should match
        ASSERT_EQ(ref_hidden_size, output_layout.spatial(0));	// input size should match

        cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

		int32_t i = 0;
		for (int32_t b = 0; b < ref_batch_size; ++b) {
			for (int32_t s = 0; s < ref_seq_len; ++s) {
				for (int32_t d = 0; d < ref_directions; ++d) {
					for (int32_t x = 0; x < ref_hidden_size; ++x) {
                        ASSERT_NEAR(ref_primitive_output[b][s][d][x], output_ptr[i++], FERROR);
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------
template<typename T>
void lstm_gpu_format_test(const cldnn::format& format, int directions, bool is_caching_test = false) {
    int layers = 1;
    int sequence_len = 6;
    int batch_size = 3;
    int input_size = 4;
    int hidden_size = 5;

    lstm_output_selection output_selection = lstm_output_selection::sequence;

    std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
            << " Sequence Len = " << sequence_len << " Directions = " << directions << " Batch Size = " << batch_size
            << " Output selection: " << static_cast<int>(output_selection) << std::endl;
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = rg.generate_random_4d<T>(1, 1, directions, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_output = VVVVF<T>(batch_size, VVVF<T>(sequence_len, VVF<T>(directions, VF<T>(hidden_size))));

    VF<T> ref_input_vec = flatten_4d<T>(format, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(cldnn::format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(cldnn::format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(cldnn::format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(format, ref_hidden);
    VF<T> ref_cell_vec = flatten_4d<T>(format, ref_cell);

    VVVVF<T> last_hidden(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));

    lstm_reference(ref_input, ref_hidden, ref_cell, ref_weights, ref_recurrent, ref_bias, ref_output,
                   last_hidden, last_cell, true, true, true,
                   (T)0, false, true);

    auto& engine = get_test_engine();

    memory::ptr input = engine.allocate_memory({ ov::element::from<T>(),format, {batch_size, sequence_len, input_size, 1} });
    memory::ptr weights = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, input_size , 4 * hidden_size } });
    memory::ptr recurrent = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, hidden_size, 4 * hidden_size } });
    memory::ptr biases = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, 1, 4 * hidden_size, directions } });
    memory::ptr hidden = engine.allocate_memory({ ov::element::from<T>(), format, { batch_size, 1, hidden_size, directions } });
    memory::ptr cell = engine.allocate_memory({ ov::element::from<T>(), format, { batch_size, 1, hidden_size, directions } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    bool emit_last_cell = output_selection == lstm_output_selection::hidden_cell ||
                          output_selection == lstm_output_selection::sequence_cell;
    bool emit_last_hidden = output_selection == lstm_output_selection::hidden ||
                            output_selection == lstm_output_selection::hidden_cell;

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<input_info> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input->get_layout()));
    for (int i = 0; i < sequence_len; ++i)
    {
        input_ids_offsets.push_back({get_string_id(i), {0, i, 0, 0}});
        lstm_inputs.push_back(input_info("inputSplit:"+get_string_id(i)));
    }
    topology.add(split("inputSplit", input_info("input"), input_ids_offsets));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    topology.add(data("biases", biases));
    topology.add(input_layout("hidden", hidden->get_layout()));
    topology.add(input_layout("cell", cell->get_layout()));
    topology.add(lstm("lstm"+get_string_id(0), lstm_inputs, "weights", "recurrent",
                      "biases", "hidden", "cell", "", 0, false,
                      { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                      output_selection, default_offset_type));

    if (emit_last_cell)
    {
        int32_t concatenation_len = emit_last_hidden ? 2 : sequence_len + 1;
        tensor hidden_tensor {batch_size, concatenation_len - 1, hidden_size, directions};
        tensor cell_tensor {batch_size, 1, hidden_size, directions};
        topology.add(crop(emit_last_hidden ? "crop:last_hidden" : "crop:sequence", input_info("lstm"), hidden_tensor, tensor{0, 0, 0, 0}));
        topology.add(crop("crop:last_cell", input_info("lstm"), cell_tensor, tensor{0, concatenation_len - 1, 0, 0}));
    }

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    std::map<primitive_id, network_output> outputs;

    network->set_input_data("input", input);
    network->set_input_data("hidden", hidden);
    network->set_input_data("cell", cell);
    outputs = network->execute();

    uint32_t ref_num_output_primitives = 1;  // Output will return atleast 1 primitive

    if (emit_last_cell) {
        // add another primitve to account for cell state if the output selection includes cell state
        ref_num_output_primitives += 1;
    }

    // check if the number of returned primitives match the expected number of output primitives
    ASSERT_EQ(ref_num_output_primitives, outputs.size());

    for (auto itr = outputs.begin(); itr != outputs.end(); itr++)
    {
        auto output_layout = itr->second.get_memory()->get_layout();
        primitive_id primitive_name = itr->first;

        cldnn::memory::ptr output_memory = itr->second.get_memory();
        int32_t output_size = (int32_t)(itr->second.get_memory()->size() / sizeof(T));
        cldnn::tensor ref_output_tensor;
        VVVVF<T> ref_primitive_output;

        int32_t ref_batch_size = batch_size;
        int32_t ref_hidden_size = hidden_size;
        int32_t ref_directions = directions;

        int32_t ref_seq_len = 1;
        // Set the reference output against which the primitive's output will be compared
        if (primitive_name.find("crop:last_cell") != std::string::npos)
        {
            ref_primitive_output = last_cell;
        }
        else if (emit_last_hidden || primitive_name.find("crop:last_hidden") != std::string::npos)
        {
            ref_primitive_output = last_hidden;
        }
        else
        {
            ref_seq_len = sequence_len;
            ref_primitive_output = ref_output;
        }

        ref_output_tensor = { ref_batch_size, ref_seq_len, ref_hidden_size, ref_directions };
        int32_t ref_output_size = ref_batch_size * ref_seq_len * ref_hidden_size * ref_directions;

        // The number of elements in reference should match the number of elements in the primitive's output
        ASSERT_EQ(ref_output_size , output_size);

        // Compare the output tensor configuration against the reference value
        // Output tensor is configured in bfyx format
        ASSERT_EQ(ref_batch_size, output_layout.batch());
        ASSERT_EQ(ref_seq_len, output_layout.feature());       // Sequence length should match
        ASSERT_EQ(ref_directions, output_layout.spatial(1));    // directions should match
        ASSERT_EQ(ref_hidden_size, output_layout.spatial(0));   // input size should match

        cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

        int32_t i = 0;
        if (format == cldnn::format::bfyx) {
            for (int32_t b = 0; b < ref_batch_size; ++b) {
                for (int32_t s = 0; s < ref_seq_len; ++s) {
                    for (int32_t d = 0; d < ref_directions; ++d) {
                        for (int32_t x = 0; x < ref_hidden_size; ++x) {
                            ASSERT_NEAR(ref_primitive_output[b][s][d][x], output_ptr[i++], FERROR);
                        }
                    }
                }
            }
        }
        else if(format == cldnn::format::fyxb)
        {
            for (int32_t s = 0; s < ref_seq_len; ++s) {
                for (int32_t d = 0; d < ref_directions; ++d) {
                    for (int32_t x = 0; x < ref_hidden_size; ++x) {
                        for (int32_t b = 0; b < ref_batch_size; ++b) {
                            ASSERT_NEAR(ref_primitive_output[b][s][d][x], output_ptr[i++], FERROR);
                        }
                    }
                }
            }
        }

    }
}

// -------------------------------------------------------
template<typename T>
void lstm_gpu_users_test(bool is_caching_test = false) {
    int sequence_len = 2;
    int batch_size = 1;
    int input_size = 1;
    int hidden_size = 1;
    int directions = 1;
    int min_random = -2, max_random = 2;
    tests::random_generator rg(GET_SUITE_NAME);

    // The following test is designed to test the user dependencies of an LSTM node when replaced by subcomponents
    // by the graph compiler.
    // The output of an LSTM node is set to last_hidden only. Then we concatenate the last_hidden with the initial_hidden tensor:
    // (input, weights, recurrent, bias, initial_hidden, inital_cell) -> LSTM -> last_hidden
    // concatenation(last_hidden, initial_hidden)
    // If the replacing is is done correctly then the initial_hidden tensor should match the output of the concatenation
    // by an offset along the sequence.

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    VVVVF<T> ref_weights = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, input_size, min_random, max_random);
    VVVVF<T> ref_recurrent = rg.generate_random_4d<T>(1, directions, 4 * hidden_size, hidden_size, min_random, max_random);
    VVVVF<T> ref_bias = rg.generate_random_4d<T>(1, 1, directions, 4 * hidden_size, min_random, max_random);
    VVVVF<T> ref_hidden = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_cell = rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random);
    VVVVF<T> ref_output = VVVVF<T>(batch_size, VVVF<T>(sequence_len, VVF<T>(directions, VF<T>(hidden_size))));

    VF<T> ref_input_vec = flatten_4d<T>(format::bfyx, ref_input);
    VF<T> ref_weights_vec = flatten_4d<T>(format::bfyx, ref_weights);
    VF<T> ref_recurrent_vec = flatten_4d<T>(format::bfyx, ref_recurrent);
    VF<T> ref_bias_vec = flatten_4d<T>(format::bfyx, ref_bias);
    VF<T> ref_hidden_vec = flatten_4d<T>(format::bfyx, ref_hidden);
    VF<T> ref_cell_vec = flatten_4d<T>(format::bfyx, ref_cell);

    VVVVF<T> last_hidden(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));
    VVVVF<T> last_cell(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))));

    auto& engine = get_test_engine();

    memory::ptr input = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {batch_size, sequence_len, input_size, 1} });
    memory::ptr weights = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, input_size , 4 * hidden_size } });
    memory::ptr recurrent = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, directions, hidden_size, 4 * hidden_size } });
    memory::ptr biases = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, 1, 4 * hidden_size, directions } });
    memory::ptr hidden = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, directions } });
    memory::ptr cell = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, directions } });

    set_values(input, ref_input_vec);
    set_values(weights, ref_weights_vec);
    set_values(recurrent, ref_recurrent_vec);
    set_values(biases, ref_bias_vec);
    set_values(hidden, ref_hidden_vec);
    set_values(cell, ref_cell_vec);

    topology topology;
    std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
    std::vector<input_info> lstm_inputs;

    topology.add(input_layout("input", input->get_layout()));
    for (int i = 0; i < sequence_len; ++i)
    {
        input_ids_offsets.push_back({get_string_id(i), {0, i, 0, 0}});
        lstm_inputs.push_back(input_info("inputSplit:"+get_string_id(i)));
    }
    topology.add(split("inputSplit", input_info("input"), input_ids_offsets));
    topology.add(data("weights", weights));
    topology.add(data("recurrent", recurrent));
    topology.add(data("biases", biases));
    topology.add(input_layout("hidden", hidden->get_layout()));
    topology.add(input_layout("cell", cell->get_layout()));
    topology.add(lstm("lstm", lstm_inputs, "weights", "recurrent",
                      "biases", "hidden", "cell", "", 0, false,
                      { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                      lstm_output_selection::hidden, default_offset_type));
    std::vector<input_info> output_ids_offsets { input_info("lstm"), input_info("hidden") };
    topology.add(concatenation("concatenation", output_ids_offsets, 1));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    std::map<primitive_id, network_output> outputs;

    network->set_input_data("input", input);
    network->set_input_data("hidden", hidden);
    network->set_input_data("cell", cell);
    outputs = network->execute();

    // check if the number of returned primitives match the expected number of output primitives
    ASSERT_EQ(size_t(1), outputs.size());
    cldnn::memory::ptr output_memory = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t s = 0; s < 1; ++s) {
            for (int32_t d = 0; d < directions; ++d) {
                for (int32_t x = 0; x < hidden_size; ++x) {
                    int32_t idx = x + hidden_size * (d + directions * ((s+1) + sequence_len * b));
                    ASSERT_NEAR(ref_hidden[b][s][d][x], output_ptr[idx], FERROR);
                }
            }
        }
    }
}

// -------------------------------------------------------
template<typename T>
void lstm_gpu_concatenated_input_test(int layers, int sequence_len, int direction,
						              int batch_size, int input_size, int hidden_size,
						              bool has_bias, bool has_initial_hidden,
						              bool has_initial_cell, float clip_threshold,
						              bool input_forget, bool is_caching_test = false)
{
    tests::random_generator rg(GET_SUITE_NAME);
	std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
		<< " Sequence Len = " << sequence_len << " Direction = " << direction << " Batch Size = " << batch_size << std::endl;
	int min_random = -2, max_random = 2;

	VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);

	std::vector<VVVVF<T>> ref_weights;
	std::vector<VVVVF<T>> ref_recurrent;
	std::vector<VVVVF<T>> ref_bias;
	std::vector<VVVVF<T>> ref_hidden;
	std::vector<VVVVF<T>> ref_cell;
	std::vector<VVVVF<T>> ref_output;

	for (int i = 0; i < layers; ++i) {
		ref_weights.push_back(rg.generate_random_4d<T>(1, direction, 4 * hidden_size, i == 0 ? input_size : hidden_size, min_random, max_random));
		ref_recurrent.push_back(rg.generate_random_4d<T>(1, direction, 4 * hidden_size, hidden_size, min_random, max_random));
		ref_bias.push_back(rg.generate_random_4d<T>(1, 1, direction, 4 * hidden_size, min_random, max_random));
		ref_hidden.push_back(rg.generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random));
		ref_cell.push_back(rg.generate_random_4d<T>(batch_size, 1, direction, hidden_size, min_random, max_random));
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

	VVVVF<T> last_hidden(batch_size, VVVF<T>(1, VVF<T>(direction, VF<T>(hidden_size))));
	VVVVF<T> last_cell(batch_size, VVVF<T>(1, VVF<T>(direction, VF<T>(hidden_size))));

	lstm_reference(ref_input, ref_hidden[0], ref_cell[0], ref_weights[0], ref_recurrent[0], ref_bias[0], ref_output[0],
		last_hidden, last_cell, has_bias, has_initial_hidden, has_initial_cell,
		clip_threshold, input_forget, true);

	for (int i = 1; i < layers; ++i) {
		lstm_reference(ref_output[i - 1], ref_hidden[i], ref_cell[i], ref_weights[i], ref_recurrent[i],
			ref_bias[i], ref_output[i],
			last_hidden, last_cell, has_bias, has_initial_hidden, has_initial_cell,
			clip_threshold, input_forget, false);
	}

	auto& engine = get_test_engine();

	memory::ptr input = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {batch_size, sequence_len, input_size, 1} });
	set_values(input, ref_input_vec);

	std::vector<memory::ptr> weights;
	std::vector<memory::ptr> recurrent;
	std::vector<memory::ptr> biases;
	std::vector<memory::ptr> hidden;
	std::vector<memory::ptr> cell;
	for (int i = 0; i < layers; ++i) {
		weights.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, direction, i == 0 ? input_size : hidden_size, 4 * hidden_size } }));
		set_values(weights[i], ref_weights_vec[i]);
		recurrent.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, direction, hidden_size, 4 * hidden_size } }));
		set_values(recurrent[i], ref_recurrent_vec[i]);
		if (has_bias) {
			biases.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, 1, 4 * hidden_size, direction } }));
			set_values(biases[i], ref_bias_vec[i]);
		}
		if (has_initial_hidden) {
			hidden.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, direction } }));
			set_values(hidden[i], ref_hidden_vec[i]);
		}
		if (has_initial_cell) {
			cell.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { batch_size, 1, hidden_size, direction} }));
			set_values(cell[i], ref_cell_vec[i]);
		}
	}

	topology topology;
	std::vector<std::pair<primitive_id, tensor>> input_ids_offsets;
	std::vector<primitive_id> lstm_inputs;
	std::vector<primitive_id> output_ids_offsets;

	topology.add(input_layout("input", input->get_layout()));
	cldnn::primitive_id prev_node_id;

    for (int i = 0; i < layers; ++i) {
		std::string sid = get_string_id(i);
		std::string lstm_id = "lstm" + sid;
		std::string weights_id = "weights" + sid;
		std::string recurrent_id = "recurrent" + sid;
		std::string biases_id = "biases" + sid;
		std::string hidden_id = "hidden" + sid;
		std::string cell_id = "cell" + sid;
		std::string output_crop_id = "crop:sequence:" + sid;

		topology.add(data(weights_id, weights[i]));
		topology.add(data(recurrent_id, recurrent[i]));
		if (has_bias) topology.add(data(biases_id, biases[i]));
		if (has_initial_hidden) topology.add(input_layout(hidden_id, hidden[i]->get_layout()));
		if (has_initial_cell) topology.add(input_layout(cell_id, cell[i]->get_layout()));
		if (i == 0) {
            topology.add(lstm(lstm_id, { input_info("input") }, weights_id, recurrent_id,
				has_bias ? biases_id : "", has_initial_hidden ? hidden_id : "", has_initial_cell ? cell_id : "", "",
				clip_threshold, input_forget,
                { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
				lstm_output_selection::sequence_cell, default_offset_type));
		}
		else {
			topology.add(lstm(lstm_id, { input_info(prev_node_id) }, weights_id, recurrent_id,
				has_bias ? biases_id : "", has_initial_hidden ? hidden_id : "", has_initial_cell ? cell_id : "", "",
				clip_threshold, input_forget,
                { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
				lstm_output_selection::sequence_cell, default_offset_type));
		}

        // Crop out the whole output sequence element
		topology.add(crop(output_crop_id, input_info(lstm_id), {batch_size, sequence_len, hidden_size, direction}, {0, 0, 0, 0}));

       // Save the node id to provide it as input to the next lstm layer
		prev_node_id = output_crop_id;
	}

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
	network->set_input_data("input", input);
	for (int i = 0; i < layers; ++i) {
		std::string sid = get_string_id(i);
		if (has_initial_hidden) network->set_input_data("hidden" + sid, hidden[i]);
		if (has_initial_cell) network->set_input_data("cell" + sid, cell[i]);
	}
	auto outputs = network->execute();
	{
		ASSERT_EQ(outputs.size(), size_t(1));
		size_t output_size = outputs.begin()->second.get_memory()->size() / sizeof(T);
		ASSERT_EQ(output_size, size_t(hidden_size * sequence_len * batch_size * direction));

		auto output = outputs.begin()->second.get_memory();

		// Get the output tensor
		cldnn::layout output_layout = output->get_layout();

		// Compare the output tensor configuration against the reference value
		// Output tensor is configured in bfyx format
		ASSERT_EQ(batch_size, output_layout.batch());
		ASSERT_EQ(sequence_len, output_layout.feature());
		ASSERT_EQ(direction, output_layout.spatial(1));
		ASSERT_EQ(hidden_size, output_layout.spatial(0));

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());
		int32_t i = 0;
		for (int32_t b = 0; b < batch_size; ++b) {
			for (int32_t s = 0; s < sequence_len; ++s) {
				for (int32_t d = 0; d < direction; ++d) {
					for (int32_t x = 0; x < hidden_size; ++x) {
						ASSERT_NEAR(ref_output[layers - 1][b][s][d][x], output_ptr[i++], FERROR);
					}
				}
			}
		}
	}
}

// This test checks chained and stacked LSTM topology. The configuration allows to create
// LSTM topology with multiple layers and can also be chained together.
template<typename T>
void lstm_gpu_chain_test(int batch_size, int input_size, int hidden_size,
                         int directions, size_t layers, size_t chains, int sequence_len,
                         const lstm_output_selection& output_selection, bool is_caching_test = false)
{
    tests::random_generator rg(GET_SUITE_NAME);
    int min_random = -2, max_random = 2;
    bool has_bias = false;
    bool has_initial_hidden = false;
    bool has_initial_cell = false;
    float clip_threshold = 0;
    bool input_forget = false;

    std::cout << "Layers = " << layers << " Input Size = " << input_size << " Hidden Size = " << hidden_size
        << " Sequence Len = " << sequence_len << " Directions = " << directions << " Batch Size = " << batch_size
        << " Output selection: " << static_cast<int>(output_selection) << std::endl;

    VVVVF<T> ref_input = rg.generate_random_4d<T>(batch_size, sequence_len, 1, input_size, min_random, max_random);
    std::vector<std::vector< VVVVF<T>>> ref_weights;
    std::vector<std::vector< VVVVF<T>>> ref_recurrent;
    std::vector<std::vector< VVVVF<T>>> ref_bias;
    std::vector<std::vector< VVVVF<T>>> ref_hidden;
    std::vector<std::vector< VVVVF<T>>> ref_cell;
    std::vector<std::vector< VVVVF<T>>> ref_output;

    // Create the 4 dimensional weight, bias, hidden, cell state and output vectors
    for (size_t chain = 0; chain < chains; chain++) {

        std::vector<VVVVF<T>> per_chain_ref_weights;
        std::vector<VVVVF<T>> per_chain_ref_recurrent;
        std::vector<VVVVF<T>> per_chain_ref_bias;
        std::vector<VVVVF<T>> per_chain_ref_hidden;
        std::vector<VVVVF<T>> per_chain_ref_cell;
        std::vector<VVVVF<T>> per_chain_ref_output;

        for (size_t layer = 0; layer < layers; layer++) {
            per_chain_ref_weights.push_back(rg.generate_random_4d<T>(1, directions, 4 * hidden_size, (layer == 0) ? input_size : hidden_size, min_random, max_random));
            per_chain_ref_recurrent.push_back(rg.generate_random_4d<T>(1, directions, 4 * hidden_size, hidden_size, min_random, max_random));
            per_chain_ref_bias.push_back(rg.generate_random_4d<T>(1, 1, directions, 4 * hidden_size, min_random, max_random));
            per_chain_ref_hidden.push_back(rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random));
            per_chain_ref_cell.push_back(rg.generate_random_4d<T>(batch_size, 1, directions, hidden_size, min_random, max_random));
            per_chain_ref_output.push_back(VVVVF<T>(batch_size, VVVF<T>(sequence_len, VVF<T>(directions, VF<T>(hidden_size)))));
        }

        ref_weights.push_back(per_chain_ref_weights);
        ref_recurrent.push_back(per_chain_ref_recurrent);
        ref_bias.push_back(per_chain_ref_bias);
        ref_hidden.push_back(per_chain_ref_hidden);
        ref_cell.push_back(per_chain_ref_cell);
        ref_output.push_back(per_chain_ref_output);
    }

    VF<T> ref_input_vec;
    std::vector<std::vector< VF<T>>> ref_weights_vec;
    std::vector<std::vector< VF<T>>> ref_recurrent_vec;
    std::vector<std::vector< VF<T>>> ref_bias_vec;
    std::vector<std::vector< VF<T>>> ref_hidden_vec;
    std::vector<std::vector< VF<T>>> ref_cell_vec;
    std::vector<std::vector< VF<T>>> ref_output_vec;

    ref_input_vec = flatten_4d<T>(cldnn::format::bfyx, ref_input);

    // flatten all the 4 dimensional vectors across chains and layers
    for (size_t chain = 0; chain < chains; chain++) {

        std::vector<VF<T>> per_chain_ref_weights;
        std::vector<VF<T>> per_chain_ref_recurrent;
        std::vector<VF<T>> per_chain_ref_bias;
        std::vector<VF<T>> per_chain_ref_hidden;
        std::vector<VF<T>> per_chain_ref_cell;
        std::vector<VF<T>> per_chain_ref_output;

        for (size_t layer = 0; layer < layers; layer++) {
            per_chain_ref_weights.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_weights[chain][layer]));
            per_chain_ref_recurrent.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_recurrent[chain][layer]));
            per_chain_ref_bias.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_bias[chain][layer]));
            per_chain_ref_hidden.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_hidden[chain][layer]));
            per_chain_ref_cell.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_cell[chain][layer]));
            per_chain_ref_output.push_back(flatten_4d<T>(cldnn::format::bfyx, ref_output[chain][layer]));
        }

        ref_weights_vec.push_back(per_chain_ref_weights);
        ref_recurrent_vec.push_back(per_chain_ref_recurrent);
        ref_bias_vec.push_back(per_chain_ref_bias);
        ref_hidden_vec.push_back(per_chain_ref_hidden);
        ref_cell_vec.push_back(per_chain_ref_cell);
        ref_output_vec.push_back(per_chain_ref_output);
    }

    std::vector<std::vector<VVVVF<T>>> last_hidden(chains, std::vector<VVVVF<T> >(layers, VVVVF<T>(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))))));
    std::vector<std::vector<VVVVF<T>>> last_cell(chains, std::vector<VVVVF<T> >(layers, VVVVF<T>(batch_size, VVVF<T>(1, VVF<T>(directions, VF<T>(hidden_size))))));

    for (size_t chain = 0; chain < chains; chain++) {
        lstm_reference(ref_input, ref_hidden[chain][0], ref_cell[chain][0], ref_weights[chain][0],
                       ref_recurrent[chain][0], ref_bias[chain][0], ref_output[chain][0],
                       last_hidden[chain][0], last_cell[chain][0], has_bias,
                       chain == 0 ? has_initial_hidden : true,
                       chain == 0 ? has_initial_cell : true,
                       clip_threshold, input_forget, true);

        if (chain < chains - 1)
        {
            ref_hidden[chain + 1][0] = last_hidden[chain][0];
            ref_cell[chain + 1][0] = last_cell[chain][0];
        }
    }

    for (size_t layer = 1; layer < layers; ++layer) {
        for (size_t chain = 0; chain < chains; chain++) {
            lstm_reference(ref_output[chain][layer - 1], ref_hidden[chain][layer], ref_cell[chain][layer],
                           ref_weights[chain][layer], ref_recurrent[chain][layer], ref_bias[chain][layer],
                           ref_output[chain][layer], last_hidden[chain][layer], last_cell[chain][layer], has_bias,
                           chain == 0 ? has_initial_hidden : true,
                           chain == 0 ? has_initial_cell : true,
                           clip_threshold, input_forget,
                           false);

            if (chain < chains - 1)
            {
                ref_hidden[chain + 1][layer] = last_hidden[chain][layer];
                ref_cell[chain + 1][layer] = last_cell[chain][layer];
            }
        }
    }

    auto& engine = get_test_engine();
    tensor input_tensor = { batch_size, sequence_len, input_size, 1 };
    layout layout = { ov::element::from<T>(), cldnn::format::bfyx, input_tensor };

    memory::ptr input = engine.allocate_memory(layout);
    set_values(input, ref_input_vec);

    // 2-dim vectors to support chain and layers
    std::vector<std::vector<memory::ptr>> weights;
    std::vector<std::vector<memory::ptr>> recurrent;
    std::vector<std::vector<memory::ptr>> biases;
    std::vector<std::vector<memory::ptr>> hidden;
    std::vector<std::vector<memory::ptr>> cell;

    for (size_t chain = 0; chain < chains; chain++) {
        std::vector<memory::ptr> per_chain_weights;
        std::vector<memory::ptr> per_chain_recurrent;
        std::vector<memory::ptr> per_chain_biases;
        std::vector<memory::ptr> per_chain_hidden;
        std::vector<memory::ptr> per_chain_cell;

        for (size_t layer = 0; layer < layers; layer++) {
            per_chain_weights.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {1, directions, layer == 0 ? input_size : hidden_size, 4 * hidden_size} }));
            set_values(per_chain_weights[layer], ref_weights_vec[chain][layer]);

            per_chain_recurrent.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {1, directions, hidden_size, 4 * hidden_size} }));
            set_values(per_chain_recurrent[layer], ref_recurrent_vec[chain][layer]);

            if (has_bias)
            {
                per_chain_biases.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {1, 1, 4 * hidden_size, directions} }));
                set_values(per_chain_biases[layer], ref_bias_vec[chain][layer]);
            }

            if (has_initial_hidden)
            {
                per_chain_hidden.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {1, 1, hidden_size, directions} }));
                set_values(per_chain_hidden[layer], ref_hidden_vec[chain][layer]);
            }

            if (has_initial_cell)
            {
                per_chain_cell.push_back(engine.allocate_memory({ ov::element::from<T>(), format::bfyx, {1, 1, hidden_size, directions} }));
                set_values(per_chain_cell[layer], ref_cell_vec[chain][layer]);
            }
        }

        weights.push_back(per_chain_weights);
        recurrent.push_back(per_chain_recurrent);
        biases.push_back(per_chain_biases);
        hidden.push_back(per_chain_hidden);
        cell.push_back(per_chain_cell);
    }

    // Start creating the topology
    cldnn::topology topology;
    std::vector<std::pair<primitive_id, cldnn::tensor>> input_ids_offsets;
    std::vector<input_info> lstm_inputs;
    std::vector<primitive_id> output_ids_offsets;

    topology.add(input_layout("input", input->get_layout()));

    for (int feature = 0; feature < sequence_len; feature++) {
        input_ids_offsets.push_back({ get_string_id(feature), {0, feature, 0, 0} });
        lstm_inputs.push_back(input_info("inputSplit:" + get_string_id(feature)));
    }
    topology.add(split("inputSplit", input_info("input"), input_ids_offsets));

    bool emit_last_hidden = output_selection == lstm_output_selection::hidden
        || output_selection == lstm_output_selection::hidden_cell;

    std::vector<cldnn::primitive_id> output_sequence_ids;
    std::vector<cldnn::primitive_id> last_hidden_ids;
    std::vector<cldnn::primitive_id> last_cell_ids;

    for (size_t chain = 0; chain < chains; chain++) {

        // Add all the primitives to the network
        std::vector<cldnn::primitive_id> prev_output_sequence_ids(output_sequence_ids);
        std::vector<cldnn::primitive_id> prev_last_hidden_ids(last_hidden_ids);
        std::vector<cldnn::primitive_id> prev_last_cell_ids(last_cell_ids);

        // Erase all the temporary primitive id containers
        output_sequence_ids.clear();
        last_cell_ids.clear();
        last_hidden_ids.clear();

        for (size_t layer = 0; layer < layers; layer++) {
            std::string chain_id = get_string_id(chain);
            std::string layer_id = get_string_id(layer);
            std::string lstm_id = "lstm:" + chain_id + ":" + layer_id;
            std::string weights_id = "weights:" + chain_id + ":" + layer_id;
            std::string recurrent_id = "recurrent:" + chain_id + ":" + layer_id;
            std::string biases_id = "biases:" + chain_id + ":" + layer_id;
            std::string hidden_id = "hidden:" + chain_id + ":" + layer_id;
            std::string cell_id = "cell:" + chain_id + ":" + layer_id;
            std::string crop_seq_id = "crop:sequence:" + chain_id + ":" + layer_id;
            std::string crop_last_cell_id = "crop:last_cell:" + chain_id + ":" + layer_id;
            std::string crop_last_hidden_id = "crop:last_hidden:" + chain_id + ":" + layer_id;

            primitive_id initial_hidden_id;
            primitive_id initial_cell_id;
            lstm_output_selection output_selection_per_layer;

            topology.add(data(weights_id, weights[chain][layer]));
            topology.add(data(recurrent_id, recurrent[chain][layer]));
            if (has_bias) topology.add(data(biases_id, biases[chain][layer]));

            if (chain == 0 && layer == 0)
            {
                if (has_initial_hidden) topology.add(input_layout(hidden_id, hidden[chain][layer]->get_layout()));
                if (has_initial_cell) topology.add(input_layout(cell_id, cell[chain][layer]->get_layout()));
            }

            // Get the initial hidden and initial cell for each layer for each chain link
            if (chain == 0)
            {
                initial_hidden_id = has_initial_hidden ? hidden_id : "";
                initial_cell_id = has_initial_cell ? cell_id : "";
            }
            else
            {
                initial_hidden_id = prev_last_hidden_ids[layer];
                initial_cell_id = prev_last_cell_ids[layer];
            }

            // Output selection for all the layers except the last layer has to have the sequence,
            // last hidden and last cell
            if (layer < layers - 1)
            {
                output_selection_per_layer = lstm_output_selection::sequence_cell;
            }
            else
            {
                // For the last layer, use the output selection provided by the user
                output_selection_per_layer = output_selection;
            }

            if (layer == 0)
            {
                topology.add(lstm(lstm_id, lstm_inputs, weights_id, recurrent_id,
                    has_bias ? biases_id : "",
                    initial_hidden_id, initial_cell_id,
                    "", clip_threshold, input_forget,
                    { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                    output_selection_per_layer, default_offset_type));
            }
            else
            {
                topology.add(lstm(lstm_id, { input_info(output_sequence_ids[layer - 1]) }, weights_id, recurrent_id,
                    has_bias ? biases_id : "",
                    initial_hidden_id, initial_cell_id,
                    "", clip_threshold, input_forget,
                    { activation_func::logistic, activation_func::hyperbolic_tan, activation_func::hyperbolic_tan }, {},
                    output_selection_per_layer, default_offset_type));
            }

            tensor sequence_tensor{ batch_size, sequence_len, hidden_size, directions };
            tensor cell_tensor{ batch_size, 1, hidden_size, directions };
            tensor last_hidden_tensor{ batch_size, 1, hidden_size, directions };

            // For all the layers except the last layer, we need to crop output sequence,
            // last hidden and last cell.
            // The output sequence goes into the next layer of lstm in a chain link
            // The last cell state and last hidden go to the lstm node in the same layer
            // next in chain
            topology.add(crop(crop_seq_id, input_info(lstm_id), sequence_tensor, tensor{ 0, 0, 0, 0 }));  // Add crop to get the sequence
            topology.add(crop(crop_last_hidden_id, input_info(lstm_id), last_hidden_tensor, tensor{ 0, sequence_len - 1, 0, 0 }));  // Add crop to get the last hidden element
            topology.add(crop(crop_last_cell_id, input_info(lstm_id), cell_tensor, tensor{ 0, sequence_len, 0, 0 }));  // Add crop to get the last cell element

            // Keep a copy of the sequence, last hidden and last cell primitve id for each layer
            output_sequence_ids.push_back(crop_seq_id);
            last_hidden_ids.push_back(crop_last_hidden_id);
            last_cell_ids.push_back(crop_last_cell_id);
        }
    }

    // Creating network out of the above designed topology
    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    for (size_t layer = 0; layer < layers; layer++) {
        std::string sid = get_string_id(layer);
        if (has_initial_hidden) network->set_input_data("hidden:000:" + sid, hidden[0][layer]); // 0 is the chain link index
        if (has_initial_cell) network->set_input_data("cell:000:" + sid, cell[0][layer]); // 0 is the chain link index
    }

    auto outputs = network->execute();
    for (auto itr = outputs.begin(); itr != outputs.end(); itr++)
    {
        auto output_layout = itr->second.get_memory()->get_layout();
        primitive_id primitive_name = itr->first;

        // Split the primitive id to get the chain id
        // Eg: primitive id: crop:last_cell:XXX:YYY
        // XXX is the chain id
        // YYY is the layer id
        std::string chain_str = primitive_name.substr(primitive_name.find(":", primitive_name.find(":") + 1) + 1, 5);
        std::string layer_str = primitive_name.substr(primitive_name.find(":", primitive_name.find(":", primitive_name.find(":") + 1) + 1) + 1, 5);
        size_t chain_id = stoi(chain_str);
        size_t layer_id = stoi(layer_str);

        cldnn::memory::ptr output_memory = itr->second.get_memory();
        int32_t output_size = (int32_t)(itr->second.get_memory()->size() / sizeof(T));
        cldnn::tensor ref_output_tensor;
        VVVVF<T> ref_primitive_output;

        int32_t ref_batch_size = batch_size;
        int32_t ref_hidden_size = hidden_size;
        int32_t ref_directions = directions;

        int32_t ref_seq_len = 1;

        // Set the reference output against which the primitive's output will be compared
        if (primitive_name.find("crop:last_cell") != std::string::npos)
        {
            ref_primitive_output = last_cell[chain_id][layer_id];
        }
        else if (emit_last_hidden || primitive_name.find("crop:last_hidden") != std::string::npos)
        {
            ref_primitive_output = last_hidden[chain_id][layer_id];
        }
        else
        {
            ref_seq_len = sequence_len;
            ref_primitive_output = ref_output[chain_id][layers - 1];
        }

        ref_output_tensor = { ref_batch_size, ref_seq_len, ref_hidden_size, ref_directions };
        int32_t ref_output_size = ref_batch_size * ref_seq_len * ref_hidden_size * ref_directions;

        // The number of elements in reference should match the number of elements in the primitive's output
        ASSERT_EQ(ref_output_size, output_size);

        // Compare the output tensor configuration against the reference value
        // Output tensor is configured in bfyx format
        ASSERT_EQ(ref_batch_size, output_layout.batch());
        ASSERT_EQ(ref_seq_len, output_layout.feature());		// Sequence length should match
        ASSERT_EQ(ref_directions, output_layout.spatial(1));	// directions should match
        ASSERT_EQ(ref_hidden_size, output_layout.spatial(0));	// input size should match

        cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

        int32_t i = 0;
        for (int32_t b = 0; b < ref_batch_size; ++b) {
            for (int32_t s = 0; s < ref_seq_len; ++s) {
                for (int32_t d = 0; d < ref_directions; ++d) {
                    for (int32_t x = 0; x < ref_hidden_size; ++x) {
                        ASSERT_NEAR(ref_primitive_output[b][s][d][x], output_ptr[i++], FERROR);
                    }
                }
            }
        }
    }
}
}  // namespace

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

// LSTM GEMM tests to test LSTM GEMMV kernel implementation
TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_test_f32) {
    generic_lstm_gemm_gpu_test<float>(5, 1, 1, 1024, 1024, true, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 256, 2, false, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_hidden_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 64, 2, true, false);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_hidden_bias_f32) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 64, 2, false, false);
}

// LSTM ELT Tests
TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.3f, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_input_forget_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.f, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_input_forget_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.5f, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.f, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f32) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, false, 0.f, false);
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
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, false, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, false, 0, false);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 0);
}

TEST(lstm_gpu, generic_lstm_input_forget_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.f, 1);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_input_forget_f32) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 1);
}

TEST(lstm_gpu, generic_lstm_offset_order_ifoz_f32) {
    default_offset_type = lstm_weights_order::ifoz;
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0, false);
    default_offset_type = lstm_weights_order::iofz;
}

TEST(lstm_gpu, generic_lstm_canonical_f32) {
    generic_lstm_gpu_test<float>(1, 1, 1, 1, 1, 1, true, true, true, 0, false);
}

// bidirectional support
TEST(lstm_gpu, generic_lstm_bi_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, false, false, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, false, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_cell_f32) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, true, 0, false);
}

// multi-layer support
TEST(lstm_gpu, generic_lstm_stacked_no_seq_f32) {
    generic_lstm_gpu_test<float>(4, 1, 1, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_f32) {
    generic_lstm_gpu_test<float>(4, 7, 1, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_stacked_bi_f32) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_bi_f32) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true, 0, false);
}

// optional outputs support
TEST(lstm_gpu, output_test_sequence_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence, 1);
}

TEST(lstm_gpu, output_test_hidden_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden, 1);
}

TEST(lstm_gpu, output_test_hidden_cell_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden_cell, 1);
}

TEST(lstm_gpu, output_test_sequence_cell_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence_cell, 1);
}

TEST(lstm_gpu, output_test_sequence_bi_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence, 2);
}

TEST(lstm_gpu, output_test_hidden_bi_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden, 2);
}

TEST(lstm_gpu, output_test_hidden_cell_bi_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden_cell, 2);
}

TEST(lstm_gpu, output_test_sequence_cell_bi_f32) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence_cell, 2);
}

// format tests
TEST(lstm_gpu, lstm_gpu_format_bfyx_f32) {
    lstm_gpu_format_test<float>(cldnn::format::bfyx, 1);
}

TEST(lstm_gpu, lstm_gpu_format_bfyx_bi_f32) {
    lstm_gpu_format_test<float>(cldnn::format::bfyx, 2);
}

TEST(lstm_gpu, lstm_gpu_format_fyxb_f32) {
    lstm_gpu_format_test<float>(cldnn::format::fyxb, 1);
}

TEST(lstm_gpu, lstm_gpu_format_fyxb_bi_f32) {
    lstm_gpu_format_test<float>(cldnn::format::fyxb, 2);
}

// test for LSTM users' dependencies
TEST(lstm_gpu, lstm_users_f32) {
    lstm_gpu_users_test<float>();
}

// Test for LSTM with concatenated input
TEST(lstm_gpu, generic_lstm_concatenated_input) {
    lstm_gpu_concatenated_input_test<float>(1, 2, 2, 1, 1, 1, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_concatenated_input_multi_layer) {
    lstm_gpu_concatenated_input_test<float>(5, 5, 2, 1, 1, 4, true, true, true, 0, false);
}

// test for LSTM with chain and stack (multilayer)
TEST(lstm_gpu, generic_lstm_chained_unidirectional_f32) {
    // batch size = 1
    // input size = 2
    // hidden size = 4
    // directions = 1
    // layers = 1
    // chains = 1
    // sequence length = 1
    // output selection = output sequence and cell
    lstm_gpu_chain_test<float>(1, 2, 4, 1, 1, 2, 1, lstm_output_selection::sequence_cell);
}

TEST(lstm_gpu, generic_lstm_chained_bidirectional_f32) {
    // batch size = 1
    // input size = 2
    // hidden size = 4
    // directions = 2
    // layers = 1
    // chains = 1
    // sequence length = 1
    // output selection = output sequence and cell
    lstm_gpu_chain_test<float>(1, 2, 4, 2, 1, 1, 1, lstm_output_selection::sequence_cell);
}

TEST(lstm_gpu, generic_lstm_chained_no_stack_bidirectional_f32) {
    // batch size = 2
    // input size = 2
    // hidden size = 4
    // directions = 2
    // layers = 1
    // chains = 2
    // sequence length = 5
    // output selection = output sequence and cell
    lstm_gpu_chain_test<float>(2, 2, 4, 2, 1, 2, 5, lstm_output_selection::sequence_cell);
}

TEST(lstm_gpu, generic_lstm_chained_stacked_bidirectional_f32) {
    // batch size = 2
    // input size = 2
    // hidden size = 4
    // directions = 2
    // layers = 4
    // chains = 2
    // sequence length = 5
    // output selection = output sequence and cell
    lstm_gpu_chain_test<float>(2, 2, 4, 2, 4, 2, 5, lstm_output_selection::sequence_cell);
}

// FP16 Half precision tests
TEST(lstm_gemm_gpu, generic_lstm_gemm_test_f16) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_bias_f16) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, false, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_f16) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, true, false);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_bias_f16) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, false, false);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_f16) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.3f, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_input_forget_f16) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.f, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_input_forget_f16) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.5f, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f16) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.f, false);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f16) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, false, 0.f, false);
}

TEST(lstm_gpu, generic_lstm_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, false, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, false, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, false, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_cell_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, false, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, false, false, 0, false);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 0);
}

TEST(lstm_gpu, generic_lstm_input_forget_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.f, 1);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_input_forget_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 1);
}

TEST(lstm_gpu, generic_lstm_offset_order_ifoz_f16) {
    default_offset_type = lstm_weights_order::ifoz;
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0, false);
    default_offset_type = lstm_weights_order::iofz;
}

TEST(lstm_gpu, generic_lstm_canonical_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 1, 1, 1, 1, 1, true, true, true, 0, false);
}

// bidirectional support
TEST(lstm_gpu, generic_lstm_bi_bias_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, false, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, true, false, 0, false);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_cell_f16) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, true, true, 0, false);
}

// multi-layer support
TEST(lstm_gpu, generic_lstm_stacked_seq_f16) {
    generic_lstm_gpu_test<ov::float16>(4, 7, 1, 3, 3, 2, true, true, true, 0, false);
}

TEST(lstm_gpu, generic_lstm_stacked_bi_f16) {
    generic_lstm_gpu_test<ov::float16>(4, 7, 2, 3, 3, 2, true, true, true, 0, false);
}

// TODO: Add tests for the following:
// integration testing using multi-layer and chained LSTMs
// LSTMs single input
// optional activation list

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(lstm_gemm_gpu, generic_lstm_gemm_test_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_bias_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, true, false, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_bias_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 3, 6, 2, false, false, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_test_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(5, 1, 1, 1024, 1024, true, true, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_bias_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 256, 2, false, true, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_hidden_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 64, 2, true, false, true);
}

TEST(lstm_gemm_gpu, gemv_bfyx_1x64_lstm_gemm_no_hidden_bias_f32_cached) {
    generic_lstm_gemm_gpu_test<float>(1, 1, 1, 64, 2, false, false, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_f32_cached) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.3f, false, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_input_forget_f32_cached) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.f, true, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_input_forget_f32_cached) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.5f, true, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f32_cached) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, true, 0.f, false, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f32_cached) {
    generic_lstm_elt_gpu_test<float>(1, 1, 4, 6, 3, false, 0.f, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_biasf32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, true, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_cell_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, true, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_cell_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, true, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_hidden_cell_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, true, false, false, true);
}

TEST(lstm_custom_gpu, generic_lstm_custom_no_bias_hidden_cell_f32_cached) {
    generic_lstm_custom_gpu_test<float>(3, 1, 3, 3, 2, false, false, false, true);
}

TEST(lstm_gpu, generic_lstm_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, false, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_cell_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, true, false, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 5, 4, 3, false, false, false, 0, false, true);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 0, true);
}

TEST(lstm_gpu, generic_lstm_input_forget_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.f, 1, true);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_input_forget_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 1, true);
}

TEST(lstm_gpu, generic_lstm_offset_order_ifoz_f32_cached) {
    default_offset_type = lstm_weights_order::ifoz;
    generic_lstm_gpu_test<float>(1, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
    default_offset_type = lstm_weights_order::iofz;
}

TEST(lstm_gpu, generic_lstm_canonical_f32_cached) {
    generic_lstm_gpu_test<float>(1, 1, 1, 1, 1, 1, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, false, false, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_bias_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, false, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_cell_f32_cached) {
    generic_lstm_gpu_test<float>(1, 7, 2, 2, 3, 4, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_stacked_no_seq_f32_cached) {
    generic_lstm_gpu_test<float>(4, 1, 1, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_f32_cached) {
    generic_lstm_gpu_test<float>(4, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_stacked_bi_f32_cached) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_bi_f32_cached) {
    generic_lstm_gpu_test<float>(4, 7, 2, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, output_test_sequence_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence, 1, true);
}

TEST(lstm_gpu, output_test_hidden_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden, 1, true);
}

TEST(lstm_gpu, output_test_hidden_cell_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden_cell, 1, true);
}

TEST(lstm_gpu, output_test_sequence_cell_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence_cell, 1, true);
}

TEST(lstm_gpu, output_test_sequence_bi_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence, 2, true);
}

TEST(lstm_gpu, output_test_hidden_bi_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden, 2, true);
}

TEST(lstm_gpu, output_test_hidden_cell_bi_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::hidden_cell, 2, true);
}

TEST(lstm_gpu, output_test_sequence_cell_bi_f32_cached) {
    lstm_gpu_output_test<float>(lstm_output_selection::sequence_cell, 2, true);
}

TEST(lstm_gpu, lstm_gpu_format_bfyx_f32_cached) {
    lstm_gpu_format_test<float>(cldnn::format::bfyx, 1, true);
}

TEST(lstm_gpu, lstm_gpu_format_bfyx_bi_f32_cached) {
    lstm_gpu_format_test<float>(cldnn::format::bfyx, 2, true);
}

TEST(lstm_gpu, lstm_gpu_format_fyxb_f32_cached) {
    lstm_gpu_format_test<float>(cldnn::format::fyxb, 1, true);
}

TEST(lstm_gpu, lstm_gpu_format_fyxb_bi_f32_cached) {
    lstm_gpu_format_test<float>(cldnn::format::fyxb, 2, true);
}

TEST(lstm_gpu, lstm_users_f32_cached) {
    lstm_gpu_users_test<float>(true);
}

TEST(lstm_gpu, generic_lstm_concatenated_input_cached) {
    lstm_gpu_concatenated_input_test<float>(1, 2, 2, 1, 1, 1, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_concatenated_input_multi_layer_cached) {
    lstm_gpu_concatenated_input_test<float>(5, 5, 2, 1, 1, 4, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_chained_unidirectional_f32_cached) {
    lstm_gpu_chain_test<float>(1, 2, 4, 1, 1, 2, 1, lstm_output_selection::sequence_cell, true);
}

TEST(lstm_gpu, generic_lstm_chained_bidirectional_f32_cached) {
    lstm_gpu_chain_test<float>(1, 2, 4, 2, 1, 1, 1, lstm_output_selection::sequence_cell, true);
}

TEST(lstm_gpu, generic_lstm_chained_no_stack_bidirectional_f32_cached) {
    lstm_gpu_chain_test<float>(2, 2, 4, 2, 1, 2, 5, lstm_output_selection::sequence_cell, true);
}

TEST(lstm_gpu, generic_lstm_chained_stacked_bidirectional_f32_cached) {
    lstm_gpu_chain_test<float>(2, 2, 4, 2, 4, 2, 5, lstm_output_selection::sequence_cell, true);
}

// FP16 Half precision tests
TEST(lstm_gemm_gpu, generic_lstm_gemm_test_f16_cached) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, true, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_bias_f16_cached) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, false, true, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_f16_cached) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, true, false, true);
}

TEST(lstm_gemm_gpu, generic_lstm_gemm_no_hidden_bias_f16_cached) {
    generic_lstm_gemm_gpu_test<ov::float16>(1, 1, 3, 6, 2, false, false, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_f16_cached) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.3f, false, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_input_forget_f16_cached) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.f, true, true);
}

TEST(DISABLED_lstm_elt_gpu, generic_lstm_elt_test_clip_input_forget_f16_cached) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.5f, true, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_test_f16_cached) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, true, 0.f, false, true);
}

TEST(lstm_elt_gpu, generic_lstm_elt_no_cell_f16_cached) {
    generic_lstm_elt_gpu_test<ov::float16>(1, 1, 4, 6, 3, false, 0.f, false, true);
}

TEST(lstm_gpu, generic_lstm_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, false, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, false, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, false, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_cell_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_cell_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_hidden_cell_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, true, false, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_no_bias_hidden_cell_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 5, 4, 3, false, false, false, 0, false, true);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 0, true);
}

TEST(DISABLED_lstm_gpu, generic_lstm_input_forget_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.f, 1, true);
}

TEST(DISABLED_lstm_gpu, generic_lstm_clip_input_forget_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0.3f, 1, true);
}

TEST(lstm_gpu, generic_lstm_offset_order_ifoz_f16_cached) {
    default_offset_type = lstm_weights_order::ifoz;
    generic_lstm_gpu_test<ov::float16>(1, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
    default_offset_type = lstm_weights_order::iofz;
}

TEST(lstm_gpu, generic_lstm_canonical_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 1, 1, 1, 1, 1, true, true, true, 0, false, true);
}

// bidirectional support
TEST(lstm_gpu, generic_lstm_bi_bias_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, false, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, true, false, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_bi_bias_hidden_cell_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(1, 7, 2, 2, 3, 4, true, true, true, 0, false, true);
}

TEST(lstm_gpu, generic_lstm_stacked_seq_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(4, 7, 1, 3, 3, 2, true, true, true, 0, false, true);
}
#endif
TEST(lstm_gpu, generic_lstm_stacked_bi_f16_cached) {
    generic_lstm_gpu_test<ov::float16>(4, 7, 2, 3, 3, 2, true, true, true, 0, false, true);
}
