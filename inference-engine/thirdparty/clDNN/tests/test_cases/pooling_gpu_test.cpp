// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/pooling.hpp>
#include <cldnn/primitives/mutable_data.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/data.hpp>

using namespace cldnn;
using namespace ::tests;

namespace cldnn {
template <>
struct type_to_data_type<FLOAT16> {
    static const data_types value = data_types::f16;
};
}  // namespace cldnn


template <typename InputT, pooling_mode Mode>
struct pooling_mode_output {
    using type = InputT;
};

template <>
struct pooling_mode_output<int8_t, pooling_mode::average> {
    using type = float;
};

template <>
struct pooling_mode_output<uint8_t, pooling_mode::average> {
    using type = float;
};

template <>
struct pooling_mode_output<int8_t, pooling_mode::average_no_padding> {
    using type = float;
};

template <>
struct pooling_mode_output<uint8_t, pooling_mode::average_no_padding> {
    using type = float;
};

template <typename InputT, pooling_mode Mode>
struct pooling_accumulator {
    static_assert(sizeof(InputT) == 0, "Input type and pooling_mode combination is not specialized");

    using output_t = typename pooling_mode_output<InputT, Mode>::type;

    void accumulate(const InputT& val);
    output_t get(size_t pool_x, size_t pool_y);
    void reset();
};

template <typename InputT>
struct pooling_accumulator<InputT, pooling_mode::max> {
    using output_t = typename pooling_mode_output<InputT, pooling_mode::max>::type;

    pooling_accumulator() : _acc(std::numeric_limits<InputT>::lowest()) {}

    void accumulate(const InputT& val) {
        using std::max;
        _acc = max(_acc, val);
    }

    output_t get(size_t /*pool_x*/, size_t /*pool_y*/, size_t /*pool_z*/) {
        return static_cast<output_t>(_acc);
    }

    void reset() { _acc = std::numeric_limits<InputT>::lowest(); }

    InputT _acc;
};

template <typename InputT>
struct pooling_accumulator<InputT, pooling_mode::average_no_padding> {
    using output_t = typename pooling_mode_output<InputT, pooling_mode::average_no_padding>::type;

    pooling_accumulator() : _acc(0), _cnt(0) {}

    void accumulate(const InputT& val) {
        _acc += static_cast<output_t>(val);
    }

    output_t get(size_t /*pool_x*/, size_t /*pool_y*/, size_t /*pool_z*/) {
        return _acc / _cnt;
    }

    void reset() {
        _acc = static_cast<output_t>(0);
        _cnt = 0;
    }

    output_t _acc;
    int _cnt;
};

template <typename InputT>
struct pooling_accumulator<InputT, pooling_mode::average> {
    using output_t = typename pooling_mode_output<InputT, pooling_mode::average>::type;

    pooling_accumulator() : _acc(0) {}

    void accumulate(const InputT& val) {
        _acc += static_cast<output_t>(val);
    }

    output_t get(size_t pool_x, size_t pool_y, size_t pool_z) {
        return static_cast<output_t>(_acc / static_cast<output_t>(pool_x * pool_y * pool_z));
    }

    void reset() {
        _acc = static_cast<output_t>(0);
    }

    output_t _acc;
};

template <typename InputT, pooling_mode Mode>
VVVF<typename pooling_mode_output<InputT, Mode>::type> reference_pooling(const VVVF<InputT>& input,
                                                                         size_t pool_x,
                                                                         size_t pool_y,
                                                                         size_t pool_z,
                                                                         int stride_x,
                                                                         int stride_y,
                                                                         int stride_z,
                                                                         int offset_x,
                                                                         int offset_y,
                                                                         int offset_z,
                                                                         bool global_pooling) {
    using output_t = typename pooling_mode_output<InputT, Mode>::type;
    VVVF<output_t> result;
    auto size_x = input[0][0].size();
    auto size_y = input[0].size();
    auto size_z = input.size();
    if (global_pooling) {
        pool_z = size_z;
        pool_y = size_y;
        pool_x = size_x;
    }

    auto accumulator = pooling_accumulator<InputT, Mode>();

    for (int zi = offset_z; zi + static_cast<int>(pool_z) <= static_cast<int>(size_z) - offset_z; zi += stride_z) {
        VVF<output_t> result_matrix;
        for (int yi = offset_y; yi + static_cast<int>(pool_y) <= static_cast<int>(size_y) - offset_y; yi += stride_y) {
            VF<output_t> result_row;
            for (int xi = offset_x; xi + static_cast<int>(pool_x) <= static_cast<int>(size_x) - offset_x; xi += stride_x) {
                accumulator.reset();
                for (int fzi = 0; fzi < static_cast<int>(pool_z); ++fzi) {
                    int index_z = zi + fzi;
                    if (index_z < 0 || index_z >= static_cast<int>(size_z))
                        continue;
                    for (int fyi = 0; fyi < static_cast<int>(pool_y); ++fyi) {
                        int index_y = yi + fyi;
                        if (index_y < 0 || index_y >= static_cast<int>(size_y))
                            continue;
                        for (int fxi = 0; fxi < static_cast<int>(pool_x); ++fxi) {
                            int index_x = xi + fxi;
                            if (index_x < 0 || index_x >= static_cast<int>(size_x))
                                continue;

                            auto input_val = input[static_cast<size_t>(index_z)][static_cast<size_t>(index_y)][static_cast<size_t>(index_x)];
                            accumulator.accumulate(input_val);
                        }
                    }
                }
                result_row.push_back(accumulator.get(pool_x, pool_y, pool_z));
            }
            result_matrix.emplace_back(std::move(result_row));
        }
        result.emplace_back(std::move(result_matrix));
    }
    return result;
}

template <typename T>
VVVF<T> reference_scale_post_op(const VVVF<T>& input, const T& scale, const T& shift) {
    auto output = input;
    auto size_z = input.size();
    auto size_y = input[0].size();
    auto size_x = input[0][0].size();
    for (size_t zi = 0; zi < size_z; ++zi) {
        for (size_t yi = 0; yi < size_y; ++yi) {
            for (size_t xi = 0; xi < size_x; ++xi) {
                output[zi][yi][xi] = output[zi][yi][xi] * scale + shift;
            }
        }
    }
    return output;
}

TEST(pooling_forward_gpu, basic_max_byxf_f32_wsiz3x3_wstr1x1_i1x3x3x8_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [ 0.5, -0.5, -0.5, -0.5, 0.5f, -0.5, -0.5f, -0.5 ]
    //  [ 1.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 2.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 3.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 4.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 5.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 6.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 7.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 8.0, 0.0, 0.0, 4.0, 0.5, -0.5, -0.5, -0.5 ]
    //
    //  Expected output:
    //  [ 8.0, 0.0, 0.0, 4,0, 0,5, -0.5, -0.5, -0.5 ]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,  format::byxf,{ 1, 8, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,3,3 }, { 1,1,1,1 }));
    network network(engine, topology);
    set_values(input_prim, { 0.5f, -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, -0.5f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        2.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        3.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        4.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        6.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        7.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        8.0f, 0.0f, 0.0f, 4.0f, 0.5f, -0.5f, -0.5f, -0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    EXPECT_EQ(4.0f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,  format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,3,3 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(2.0f, output_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_global_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool mode: max
    //  Global pooling: true
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,  format::yxfb,{ 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(2.0f, output_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_b_fs_yx_fsv16_i8_global_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool mode: max
    //  Global pooling: true
    //  Padding: none

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::i8, format::b_fs_yx_fsv16, { 1, 16, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max));

    network network(engine, topology);
    std::vector<char> vals = {
           0,  3,  2, -1,  6,   8,  3,  -9,  6, -1,  1,  7,  -1,  6,  18,  3,
          -9,  5, -2,  2,  6,  -1,  6,   7,  3, -9,  6, -3,   3,  5,  -1, 16,
           8,  3, -9,  6, -4,   4,  3,  -1,  6,  8, 33, -9,   6, -5,   5, 21,
          -1,  6,  8,  3, -9,   6, -5,  36,  2, -1,  6,  8,   3, -9,   6, -6,
           6,  1, -1,  6,  8,   3, -9,  66, -7,  7, 29, -1,   6,  8,   3, -9,
           6, 44,  8, -2, -1,   6,  8,   3, -9,  6, -8,  9,  -1, 10,   6,  8,
           3, -9,  6, -9, 10,  -3, -1,   6,  8,  3, 99,  6, -10, 11,  -4, -1,
           6,  8,  3, -9, 64, -11, 12,  -5, -1,  6,  8, 38,  -9,  6, -12, 13,
          -2, -1,  6, 81,  3,  -9,  6, -13, 14, -2, -1, 64,   8,  3,  -9,  6,
    };
    set_values(input_prim, vals);
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<char> output_ptr(output_prim, get_test_stream());

    std::vector<char> answers = { 8, 44, 8, 81, 64, 8, 12, 66, 14, 8, 99, 64, 8, 11, 18, 21 };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_avg_b_fs_yx_fsv16_i8_global_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool mode: avg
    //  Global pooling: true
    //  Padding: none

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::i8, format::b_fs_yx_fsv16, { 1, 16, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average));

    network network(engine, topology);
    std::vector<char> vals = {
           0,  3,  2, -1,  6,   8,  3,  -9,  6, -1,  1,  7,  -1,  6,  18,  3,
          -9,  5, -2,  2,  6,  -1,  6,   7,  3, -9,  6, -3,   3,  5,  -1, 16,
           8,  3, -9,  6, -4,   4,  3,  -1,  6,  8, 33, -9,   6, -5,   5, 21,
          -1,  6,  8,  3, -9,   6, -5,  36,  2, -1,  6,  8,   3, -9,   6, -6,
           6,  1, -1,  6,  8,   3, -9,  66, -7,  7, 29, -1,   6,  8,   3, -9,
           6, 44,  8, -2, -1,   6,  8,   3, -9,  6, -8,  9,  -1, 10,   6,  8,
           3, -9,  6, -9, 10,  -3, -1,   6,  8,  3, 99,  6, -10, 11,  -4, -1,
           6,  8,  3, -9, 64, -11, 12,  -5, -1,  6,  8, 38,  -9,  6, -12, 13,
          -2, -1,  6, 81,  3,  -9,  6, -13, 14, -2, -1, 64,   8,  3,  -9,  6,
    };
    set_values(input_prim, vals);
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<uint8_t> output_ptr(output_prim, get_test_stream());

    std::vector<uint8_t> answers = {
         29, 199, 241, 63,  85,  85, 213, 64,  85,  85,  21, 64, 142, 227,   8, 65,
         57, 142,  19, 65, 171, 170, 170, 62,  57, 142,  35, 64,   0,   0,  32, 65,
        199, 113,  28, 64,  29, 199, 241, 63,  29, 199, 153, 65,  57, 142,  83, 65,
        228,  56,  14, 63, 142, 227, 120, 64, 171, 170, 170, 63,  85,  85, 181, 64,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]) << i;
    }
}

TEST(pooling_forward_gpu, basic_max_pooling_int8) {

    auto& engine = get_test_engine();
    layout in_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,3,3 } };
    layout out_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,1,1 } };
    layout byte_layout = { type_to_data_type<int8_t>::value, format::bfyx,{ 1,1,3,3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.5f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.5f };
    std::list<float> final_results = { 10.0f };

    // Allocate memory for input image.
    auto input_memory = engine.allocate_memory(in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input", input, byte_layout),
        pooling("pool1", "reorder_input", pooling_mode::max, { 1,1,3,3 }, {1,1,1,1}),
        reorder("reorder2", "pool1", out_layout)
    );

    network network(
        engine,
        topology,
        build_options{
            build_option::outputs({ "reorder2" })
        });

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    unsigned int cntr = 0;
    for (const auto& exp : final_results)
    {
        EXPECT_EQ(exp, interm_ptr[cntr++]);
    }
}

TEST(pooling_forward_gpu, basic_avg_pooling_int8) {

    auto& engine = get_test_engine();
    layout in_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,3,3 } };
    layout out_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,1,1 } };
    layout byte_layout = { type_to_data_type<int8_t>::value, format::bfyx,{ 1,1,3,3 } };
    std::initializer_list<float> input_f = { 2.0f, -2.5f, 5.1f, -4.0f, 8.03f, -6.99f, 17.0f, -8.0f, 19.5f };
    // Average pooling returns fp32 by default for int8 inputs
    auto final_result = 0.0f;
    for (const auto& val : input_f)
    {
        // reorder fp32 -> int8 do round
        final_result += (float)(std::roundf(val));
    }
    final_result /= input_f.size();
    // Allocate memory for input image.
    auto input_memory = engine.allocate_memory(in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input", input, byte_layout),
        pooling("pool1", "reorder_input", pooling_mode::average, { 1,1,3,3 }, { 1,1,1,1 }),
        reorder("reorder2", "pool1", out_layout)
    );

    network network(
        engine,
        topology,
        build_options{
            build_option::outputs({ "reorder2" })
        });

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    EXPECT_EQ(final_result, interm_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0,  1.5]
    //  [ 2.0,  1.5]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(1.5f, output_ptr[1]);
    EXPECT_EQ(2.0f, output_ptr[2]);
    EXPECT_EQ(1.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.25,  1.00,  0.50,  0.25]
    //  [ 2.00,  1.50, -0.50, -0.75]
    //  [ 0.00, -1.00,  0.50,  0.25]
    //  [ 0.50, -2.00, -1.50, -2.50]
    //
    //  Expected output:
    //  [ 2.0,  0.5]
    //  [ 0.5,  0.5]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 4, 4 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }));

    network network(engine, topology);
    set_values(input_prim, { -0.25f, 1.00f, 0.50f, 0.25f, 2.00f, 1.50f, -0.50f, -0.75f, 0.00f, -1.00f, 0.50f, 0.25f, 0.50f, -2.00f, -1.50f, -2.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(0.5f, output_ptr[1]);
    EXPECT_EQ(0.5f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x2x2_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [-0.5,  0.5,  0.0]   [-1.5, -0.5,  0.0]
    //  [ 1.0, -1.0, -2.0]   [ 0.0, -1.0,  1.5]
    //  [-1.0, -0.5, -0.5]   [-2.0,  1.0, -0.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  0.0, -0.5]   [ 0.0,  0.5, -0.5]
    //  [-2.0, -1.0,  1.0]   [ 1.0, -1.0,  0.0]
    //  [-0.5, -1.0,  1.5]   [ 0.5, -0.5,  0.0]
    //
    //  Expected output:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [ 1.0,  0.5]         [ 0.0,  1.5]
    //  [ 1.0, -0.5]         [ 1.0,  1.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  1.0]         [ 1.0,  0.5]
    //  [-0.5,  1.5]         [ 1.0,  0.0]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 0.5f, -1.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f, -0.5f, 0.0f, -0.5f, 1.0f, -2.0f, 0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, 1.0f, 1.5f, 0.0f, -1.0f, -0.5f, -2.0f, 0.5f, -0.5f, -1.0f, 1.0f, -0.5f, -0.5f, 1.5f, -0.5f, 0.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(1.0f, output_ptr[0]); EXPECT_EQ(0.0f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[4]); EXPECT_EQ(1.5f, output_ptr[6]);
    EXPECT_EQ(1.0f, output_ptr[8]); EXPECT_EQ(1.0f, output_ptr[10]);
    EXPECT_EQ(-0.5f, output_ptr[12]); EXPECT_EQ(1.5f, output_ptr[14]);

    EXPECT_EQ(0.5f,  output_ptr[1]);  EXPECT_EQ(1.0f, output_ptr[3]);
    EXPECT_EQ(1.0f,  output_ptr[5]);  EXPECT_EQ(0.5f, output_ptr[7]);
    EXPECT_EQ(-0.5f, output_ptr[9]);  EXPECT_EQ(1.0f, output_ptr[11]);
    EXPECT_EQ(1.5f,  output_ptr[13]); EXPECT_EQ(0.0f, output_ptr[15]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5, -0.5]
    //  [   -1, 0.5]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0, 0, -1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.50f, -0.50f, -1.00f, 0.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    EXPECT_EQ( 1.5f, output_ptr[0]);
    EXPECT_EQ(-0.5f, output_ptr[1]);
    EXPECT_EQ(-1.0f, output_ptr[2]);
    EXPECT_EQ( 0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5,  -0.5]
    //  [   1,  -0.5]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);

    set_values(input_prim, {
        1.50f, -1.00f, -0.50f,
        1.00f, -1.00f, -1.00f,
       -1.00f, -1.00f, -0.50f
    });

    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    EXPECT_EQ((int)output_prim->get_layout().size.count(), 4);

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    EXPECT_EQ(1.5f, output_ptr[0]);
    EXPECT_EQ(-0.5f, output_ptr[1]);
    EXPECT_EQ(1.0f, output_ptr[2]);
    EXPECT_EQ(-0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_avg_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //  [ 1.0,   0.625]
    //  [ 1.625, 0.875]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average,{ 1,1,2,2 },{ 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 4.0f, -1.0f, 3.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(1.0f,   output_ptr[0]);
    EXPECT_EQ(0.625f, output_ptr[1]);
    EXPECT_EQ(1.625f, output_ptr[2]);
    EXPECT_EQ(0.875f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 0.375, -0.125]
    //  [ -0.25,  0.125]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(-0.125f, output_ptr[1]);
    EXPECT_EQ(-0.25f,  output_ptr[2]);
    EXPECT_EQ(0.125f,  output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_bfyx_f32_wsiz3x3_wstr3x3_i1x1x3x3_zeropad) {
    //  Test the corner case when average pooling window contains data from image, data from padding and data outside padding
    //
    //  Pool window: 3x3
    //  Pool stride: 3x3
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, -1.0, padd]
    //  [ padd,  0.5,  0.1,  0.2, padd]
    //  [ padd,  0.9,  1.1,  2.2, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 0.177777, -0.133333]
    //  [ 0.333333,  0.55]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,3,3 }, { 1,1,3,3 }, { 0,0,-1,-1 }));

    network network(engine, topology);

    std::vector<float> input_vec = { 1.5f, -0.5f, -1.0f, 0.5f, 0.1f, 0.2f, 0.9f, 1.1f, 2.2f };
    set_values(input_prim, input_vec);

    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_NEAR(output_ptr[0], 0.177777f, 1e-05F);
    EXPECT_NEAR(output_ptr[1], -0.133333f, 1e-05F);
    EXPECT_NEAR(output_ptr[2], 0.333333f, 1e-05F);
    EXPECT_NEAR(output_ptr[3], 0.55f, 1e-05F);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5,  2.5]
    //  [ padd, -1.0,  0.5,  3.0]
    //  [ padd,  0.5,  0.0, -8.0]
    //
    //  Expected output:
    //  [  0.375,    0.5]
    //  [ -0.125, -1.125]

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, 2.5f, -1.0f, 0.5f, 3.0f, 0.5f, 0.0f, -8.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    EXPECT_EQ((int)output_prim->get_layout().size.count(), 4);

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(0.5f,    output_ptr[1]);
    EXPECT_EQ(-0.125f, output_ptr[2]);
    EXPECT_EQ(-1.125f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, -0.5, 0, 0]
    //  [0, 1, -0.5, 0, 0]
    //  [0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f,-0.5f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim->get_layout().size.count(), 4);
        EXPECT_EQ((int)output_prim->get_layout().get_buffer_size().count(), 16);

        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim->get_layout().with_padding(padding{ {0,0,1,2}, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, -0.5, 0]
    //  [0, 1, -0.5, 0]
    //  [0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f, -0.5f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim->get_layout().size.count(), 4);
        EXPECT_EQ((int)output_prim->get_layout().get_buffer_size().count(), 16);

        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4]
    //  [ 5,  1.5, -0.5, 6]
    //  [ 7, -1.0,  0.5, 8]
    //  [ 9, 10, 11, 12]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 2.375, 3.125, 0, 0]
    //  [ 0, 0, 6.25,  7.875, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 4, 4 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim->get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,0,0 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f,
            5.f, 1.5f, -0.5f, 6.f,
            7.f, -1.0f, 0.5f, 8.f,
            9.f, 10.f, 11.f, 12.f});
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 2.375f, 3.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 6.25f, 7.875f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4, 5]
    //  [ 6,  1.5, -1.0, -0.5, 7]
    //  [ 8,  1.0, -1.0, -1.0, 9]
    //  [ 10, -1.0, -1.0, -0.5, 11]
    //  [ 12, 13, 14, 15, 16]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1, 3, 5, 0]
    //  [0, 8, 1.5, 9, 0]
    //  [0, 12, 14, 16, 0]
    //  [0, 0, 0, 0, 0]

    auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 5, 5 );
        auto input_prim = engine.allocate_memory({ data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim->get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f, 5.f,
            6.f, 1.50f, -1.00f, -0.50f, 7.f,
            8.f, 1.00f, -1.00f, -1.00f, 9.f,
            10.f, -1.00f, -1.00f, -0.50f, 11.f,
            12.f, 13.f, 14.f, 15.f, 16.f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.f, 3.f, 5.f, 0.0f,
            0.0f, 8.f, 1.5f, 9.f, 0.0f,
            0.0f, 12.f, 14.f, 16.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim->get_layout().size.count(), 9);
        EXPECT_EQ((int)output_prim->get_layout().get_buffer_size().count(), 25);

        cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2x1_max_with_argmax) {
    //  Input  : 2x2x3x2x1
    //  Argmax : 2x2x2x1x1
    //  Output : 2x2x2x2x1

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 3, 2, 1 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 2, 1, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2, 1 }, { 1, 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(output_layout.size.spatial[2], 1);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_input_padding) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2
    //  Input Padding : 2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", input->get_layout().with_padding(padding{ { 0, 0, 2, 2 }, 0 })));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "reorder", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_output_padding) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2
    //  Output Padding : 2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", input->get_layout().with_padding(padding{ { 0, 0, 2, 2 }, 0 })));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "reorder", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 8.0f, 16.0f,0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 12.0f, 17.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }

    for (size_t i = 0; i < expected_argmax_vec.size(); ++i) {
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_with_output_size) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 2, 2, 2, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

template <class DataType>
static void generic_average_wo_padding_test(format fmt, tensor output, tensor input, tensor window, tensor stride, tensor offset)
{
    constexpr auto dt = std::is_same<DataType, float>::value ? data_types::f32 : data_types::f16;

    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        if (dt == data_types::f16) {
            return;
        }
    }

    auto input_mem = engine.allocate_memory(layout{ dt, fmt, input });
    set_values(input_mem, std::vector<DataType>(input.count(), DataType(1)));
    std::vector<DataType> expected_output(output.count(), DataType(1));

    topology tpl;
    tpl.add(input_layout("in", input_mem->get_layout()));

    auto pool_in = "in";
    if (offset != tensor())
    {
        tpl.add(reorder("reorder", "in", input_mem->get_layout().with_padding((padding) offset.negate().sizes())));
        pool_in = "reorder";
    }
    tpl.add(pooling("pool", pool_in, pooling_mode::average_no_padding, window, stride, offset));

    network net(engine, tpl);
    net.set_input_data("in", input_mem);
    auto output_mem = net.execute().at("pool").get_memory();

    ASSERT_EQ(output_mem->count(), expected_output.size());
    EXPECT_EQ(output_mem->get_layout().size, output);
    cldnn::mem_lock<DataType> out_ptr(output_mem, get_test_stream());

    for (size_t i = 0; i < expected_output.size(); ++i)
        EXPECT_FLOAT_EQ(out_ptr[i], expected_output[i]);
}

//bfyx fp32
TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2)
{
    generic_average_wo_padding_test<float>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s3x3_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i1x1_w3x3_s1x1_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, (tensor) spatial(1, 1), (tensor) spatial(1, 1), (tensor) spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//bfyx fp16
TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s3x3_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i1x1_w3x3_s1x1_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, (tensor) spatial(1, 1), (tensor) spatial(1, 1), (tensor) spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//yxfb fp32
TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2)
{
    generic_average_wo_padding_test<float>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s3x3_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i1x1_w3x3_s1x1_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, (tensor) spatial(1, 1), (tensor) spatial(1, 1), (tensor) spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//yxfb fp16
TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s3x3_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, (tensor) spatial(2, 2), (tensor) spatial(3, 3), (tensor) spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i1x1_w3x3_s1x1_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, (tensor) spatial(1, 1), (tensor) spatial(1, 1), (tensor) spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//bfzyx fp32
TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x2_s2x2x2)
{
    generic_average_wo_padding_test<float>(format::bfzyx, (tensor) spatial(2, 2,  2), (tensor) spatial(3, 3, 3), (tensor) spatial(2, 2, 2), tensor{ 0,0,2,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x2_s2x2x2_o1x1x1)
{
    generic_average_wo_padding_test<float>(format::bfzyx, (tensor) spatial(2, 2, 2), (tensor) spatial(3, 3, 3), (tensor) spatial(2, 2, 3), tensor{ 0,0,2,2,3 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x2_s3x3x3_o1x1x1)
{
    generic_average_wo_padding_test<float>(format::bfzyx, (tensor) spatial(2, 2, 2), (tensor) spatial(3, 3, 3), (tensor) spatial(3, 3, 3), tensor{ 0,0,2,2,2 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i1x1x1_w3x3x3_s1x1x1_o1x1x1)
{
    generic_average_wo_padding_test<float>(format::bfzyx, (tensor) spatial(1, 1, 1), (tensor) spatial(1, 1, 1), (tensor) spatial(3, 3, 3), tensor{ 0,0,1,1,1 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w3x3x3_s3x3x3)
{
    generic_average_wo_padding_test<float>(format::bfzyx, (tensor) spatial(1, 1, 1), (tensor) spatial(3, 3, 3), (tensor) spatial(3, 3, 3), tensor{ 0,0,3,3,3 }, tensor{});
}

//bfzyx fp16
TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x2_s2x2x2_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfzyx, (tensor) spatial(2, 2, 2), (tensor) spatial(3, 3, 3), (tensor) spatial(2, 2, 2), tensor{ 0,0,2,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x2_s2x2x2_o1x1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfzyx, (tensor) spatial(2, 2, 2), (tensor) spatial(3, 3, 3), (tensor) spatial(2, 2, 2), tensor{ 0,0,2,2,2 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w2x2x3_s3x3x3_o1x1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfzyx, (tensor) spatial(2, 2, 2), (tensor) spatial(3, 3, 3), (tensor) spatial(3, 3, 3), tensor{ 0,0,2,2,2 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i1x1x1_w3x3x3_s1x1x1_o1x1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfzyx, (tensor) spatial(1, 1, 1), (tensor) spatial(1, 1, 1), (tensor) spatial(3, 3, 3), tensor{ 0,0,1,1,1 }, tensor{ 0,0,-1,-1,-1 });
}

TEST(pooling_forward_gpu, bfzyx_average_without_padding_i3x3x3_w3x3x3_s3x3x3_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfzyx, (tensor) spatial(1, 1, 1), (tensor) spatial(3, 3, 3), (tensor) spatial(3, 3, 3), tensor{ 0,0,3,3,3 }, tensor{});
}

TEST(pooling_forward_gpu, b_fs_yx_fsv4)
{
    int B_array[] = {  16,    4, 0 };  // Batch
    int F_array[] = {  64, 2048, 0 };  // Features
    int I_array[] = { 112,    7, 0 };  // Input MxM data sizes
    int W_array[] = {   7,    3, 0 };  // Filter (a-ka weights) sizes
    int S_array[] = {   1,    2, 0 };  // Strides
    for (int j = 0; F_array[j]; j++) {
        int in_B = B_array[j];

        int in_F = F_array[j];

        int in_X = I_array[j],
            in_Y = in_X;

        int W_X = W_array[j],
            W_Y = W_X;

        int S_X = S_array[j],
            S_Y = S_X;

        // Input data init
        std::vector<char> Data(in_B * in_F * in_X * in_Y);
        for (size_t i = 0; i < Data.size(); i++)
            Data[i] = static_cast<char>(i);
        std::vector<char> DataGold(Data);

        // Expected "gold" output and IMAD output.
        std::vector<char>  vGoldOutput;
        std::vector<char>  vTestOutput;

        auto& engine = get_test_engine();

        // "Golden" Pooling
        {
            // Mem initialization
            // This is user data, no kernels here
            auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
            set_values(input, std::move(DataGold));

            auto pool = pooling("pool_GOLD",
                                 "input",
                                 pooling_mode::max,
                                 { 1, 1, W_X, W_Y },  // kernel_size
                                 { 1, 1, S_X, S_Y }); // stride

            // Create a topology with a simple Convolution layer
            topology topology(input_layout("input", input->get_layout()),
                              pool);

            // Network processing
            network network(engine, topology);
            network.set_input_data("input", input);
            //network_exe(network, vGoldOutput, "pool_GOLD");
            auto outputs = network.execute();
            auto searchC = outputs.find("pool_GOLD");
            ASSERT_FALSE(searchC == outputs.end());
            auto output = outputs.begin()->second.get_memory();
            cldnn::mem_lock<char> output_ptr(output, get_test_stream());
            vGoldOutput.reserve(output_ptr.size());
            for (size_t i = 0; i < output_ptr.size(); i++)
                vGoldOutput.push_back(output_ptr[i]);
        }

        //
        // IMAD Pooling
        //
        {
            topology topology;

            // Mem initialization
            // This is user data, no kernels here
            auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
            set_values(input, std::move(Data));

            // Add input to topology
            topology.add(
                input_layout("input", input->get_layout()));

            // Reorder (a-ka swizzelling) input to MMAD/IMAD Pooling format
            topology.add(reorder("reorder_Swizzelled",
                         "input",
                         layout(data_types::i8,
                                format::b_fs_yx_fsv4,
                                { in_B, in_F, in_X, in_Y })));

            // Add Convoluiton to topology
            topology.add(pooling("pool_IMAD",
                                 "reorder_Swizzelled",
                                 pooling_mode::max,
                                 { 1, 1, W_X, W_Y },  // kernel_size
                                 { 1, 1, S_X, S_Y })); // stride

            // Back reordering (a-ka unswizzelling) output from MMAD/IMAD pooling
            topology.add(reorder("reorder_UnSwizzelled",
                                 "pool_IMAD",
                                 layout(data_types::i8,
                                        format::bfyx,
                                        { in_B, in_F, in_X, in_Y })));

            network network(engine, topology);
            network.set_input_data("input", input);
            //network_exe(network, vTestOutput, "reorder_UnSwizzelled");
            auto outputs = network.execute();
            auto searchC = outputs.find("reorder_UnSwizzelled");
            ASSERT_FALSE(searchC == outputs.end());
            auto output = outputs.begin()->second.get_memory();
            cldnn::mem_lock<char> output_ptr(output, get_test_stream());
            vTestOutput.reserve(output_ptr.size());
            for (size_t i = 0; i < output_ptr.size(); i++)
                vTestOutput.push_back(output_ptr[i]);
        }

        // Result validation
        ASSERT_TRUE(vGoldOutput.size() == vTestOutput.size());
        for (size_t i = 0; i < vGoldOutput.size(); i++)
            ASSERT_TRUE(vTestOutput[i] == vGoldOutput[i]);

    } // for (int j = 0; F_array[j]; i++)
}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_avg_3x3_input_2x2_pool_1x1_stride_2x2_output)
{
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //  [ 1.0,   0.625]
    //  [ 1.625, 0.875]

    auto input_prim = engine.allocate_memory({ data_types::f16, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input", input_prim->get_layout()));
    topology.add(reorder("reorder_input", "input", layout(data_types::f16, format::fs_b_yx_fsv32, { 1, 1, 3, 3 })));
    topology.add(pooling("avg_pooling", "reorder_input", pooling_mode::average, { 1,1,2,2 }, { 1,1,1,1 }));
    topology.add(reorder("reorder_after_pooling", "avg_pooling", layout(data_types::f16, format::bfyx, { 1,1,2,2 })));

    network network(engine, topology);
    set_values(input_prim, { FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f) });
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_after_pooling");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    EXPECT_EQ(1.0f, float(output_ptr[0]));
    EXPECT_EQ(0.625f, float(output_ptr[1]));
    EXPECT_EQ(1.625f, float(output_ptr[2]));
    EXPECT_EQ(0.875f, float(output_ptr[3]));

}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_avg_3x3_input_2x2_pool_2x2_stride)
{
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //  [ 1.0, 0  ]
    //  [ 1.5, 3.5]

    auto input_prim = engine.allocate_memory({ data_types::f16, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input", input_prim->get_layout()));
    topology.add(reorder("reorder_input", "input", layout(data_types::f16, format::fs_b_yx_fsv32, { 1, 1, 3, 3 })));
    topology.add(pooling("avg_pooling", "reorder_input", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }));
    topology.add(reorder("reorder_after_pooling", "avg_pooling", layout(data_types::f16, format::bfyx, { 1,1,3,3 })));

    network network(engine, topology);
    set_values(input_prim, { FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f) });
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_after_pooling");

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    EXPECT_EQ(1.0f, float(output_ptr[0]));
    EXPECT_EQ(0.f, float(output_ptr[1]));

    EXPECT_EQ(1.5f, float(output_ptr[2]));
    EXPECT_EQ(3.5f, float(output_ptr[3]));

}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_avg_2x2x3x3_input_2x2_pool_2x2_stride)
{
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //              B 0                 B 1
    //      [-0.5,  1.0,  0.5]  [-0.5,  1.0,  0.5]
    //  F0  [ 2.0,  1.5, -0.5]  [ 2.0,  1.5, -0.5]
    //      [ 4.0, -1.0,  3.5]  [ 4.0, -1.0,  3.5]
    //
    //      [-0.5,  1.0,  0.5]  [-0.5,  1.0,  0.5]
    //  F1  [ 2.0,  1.5, -0.5]  [ 2.0,  1.5, -0.5]
    //      [ 4.0, -1.0,  3.5]  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //          B 0          B 1
    //      [ 1.0, 0  ]  [ 1.0, 0  ]
    //  F0  [ 1.5, 3.5]  [ 1.5, 3.5]
    //
    //      [ 1.0, 0  ]  [ 1.0, 0  ]
    //  F1  [ 1.5, 3.5]  [ 1.5, 3.5]
    //
    const int features_count = 2;
    const int batch_count = 2;
    const int out_x = 2;
    const int out_y = 2;

    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_count, features_count, 3, 3 } });

    topology topology;
    topology.add(input_layout("input", input_prim->get_layout()));
    topology.add(reorder("reorder_input", "input", layout(data_types::f16, format::fs_b_yx_fsv32, { batch_count, features_count, 3, 3 })));
    topology.add(pooling("avg_pooling", "reorder_input", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }));
    topology.add(reorder("reorder_after_pooling", "avg_pooling", layout(data_types::f16, format::bfyx, { batch_count, features_count, out_y, out_x })));

    network network(engine, topology);
    set_values(input_prim, { FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f),   //B0F0
                             FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f),   //B0F1
                             FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f),   //B1F0
                             FLOAT16(-0.5f), FLOAT16(1.0f), FLOAT16(0.5f), FLOAT16(2.0f), FLOAT16(1.5f), FLOAT16(-0.5f), FLOAT16(4.0f), FLOAT16(-1.0f), FLOAT16(3.5f) });//B1F1
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_after_pooling");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    ASSERT_EQ((int)output_ptr.size(), batch_count * features_count*out_x*out_y);


    for (int b = 0; b<batch_count; b++)
    {
        for (int f = 0; f < features_count; f++)
        {
            const int f_pitch = out_x * out_y;
            const int bf_offset = b * (f_pitch * features_count) + f * f_pitch;
            EXPECT_EQ(1.0f, float(output_ptr[bf_offset + 0])); // X0Y0
            EXPECT_EQ(0.f,  float(output_ptr[bf_offset + 1])); // X1Y0

            EXPECT_EQ(1.5f, float(output_ptr[bf_offset + 2])); // X0Y1
            EXPECT_EQ(3.5f, float(output_ptr[bf_offset + 3])); // X1Y1
        }
    }
}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_max_1x1x3x3_input_2x2_pool_2x2_stride_2x2_outpad) {
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0,    0,    0,  0,  0]
    //  [0,  1.5, -0.5,  0,  0]
    //  [0,    1, -0.5,  0,  0]
    //  [0,    0,    0,  0,  0]

        tensor input_tensor(1, 1, 3, 3);
        auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim->get_layout()));
        topology.add(reorder("reorder_input", "input_prim", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
        topology.add(pooling("pool_prim", "reorder_input", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));
        topology.add(reorder("reorder_pooling", "pool_prim", layout(data_types::f16, format::bfyx, { 1,1,4,4 }, padding{ {0,0,1,1},0 })));

        network network(engine, topology);

        set_values(input_prim, {
            FLOAT16(1.50f), FLOAT16(-1.00f), FLOAT16(-0.50f),
            FLOAT16(1.00f), FLOAT16(-1.00f), FLOAT16(-1.00f),
            FLOAT16(-1.00f), FLOAT16(-1.00f), FLOAT16(-0.50f)
            });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f,-0.5f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "reorder_pooling");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim->get_layout().size.count(), 4);
        EXPECT_EQ((int)output_prim->get_layout().get_buffer_size().count(), 16);

        cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], float(output_ptr[i]));
        }

}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_max_1x1x5x5_input_2x2_pool_2x2_stride_2x2_outpad_2x1_inpad) {
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 1x1
    //  Input data:
    //  [ 1,  2,     3,    4,   5 ]
    //  [ 6,  1.5,  -1.0, -0.5, 7 ]
    //  [ 8,  1.0,  -1.0, -1.0, 9 ]
    //  [ 10, -1.0, -1.0, -0.5, 11]
    //  [ 12, 13,    14,   15,  16]
    //
    //  Expected output:
    //  [ 0,  0,  0,    0,  0]
    //  [ 0,  1,  3,    5,  0]
    //  [ 0,  8,  1.5,  9,  0]
    //  [ 0, 12,  14,  16,  0]
    //  [ 0,  0,  0,    0,  0]

    tensor input_tensor(1, 1, 5, 5);
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, input_tensor });

    topology topology;
    topology.add(input_layout("input_prim", input_prim->get_layout()));
    topology.add(reorder("reorder_input", "input_prim", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ { 0,0,2,1 } , 0 })));
    topology.add(pooling("pool_prim", "reorder_input", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));
    topology.add(reorder("reorder_pooling", "pool_prim", layout(data_types::f16, format::bfyx, input_tensor, padding{{0,0,1,1},0})));

    network network(engine, topology);

    set_values(input_prim, {
        FLOAT16(1.f),  FLOAT16(2.f),    FLOAT16(3.f),    FLOAT16(4.f),    FLOAT16(5.f),
        FLOAT16(6.f),  FLOAT16(1.50f),  FLOAT16(-1.00f), FLOAT16(-0.50f), FLOAT16(7.f),
        FLOAT16(8.f),  FLOAT16(1.00f),  FLOAT16(-1.00f), FLOAT16(-1.00f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(-1.00f), FLOAT16(-1.00f), FLOAT16(-0.50f), FLOAT16(11.f),
        FLOAT16(12.f), FLOAT16(13.f),   FLOAT16(14.f),   FLOAT16(15.f),   FLOAT16(16.f)
        });

    network.set_input_data("input_prim", input_prim);

    std::vector<float> expected = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.f, 3.f, 5.f, 0.0f,
        0.0f, 8.f, 1.5f, 9.f, 0.0f,
        0.0f, 12.f, 14.f, 16.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_pooling");

    auto output_prim = outputs.begin()->second.get_memory();
    EXPECT_EQ((int)output_prim->get_layout().size.count(), 9);
    EXPECT_EQ((int)output_prim->get_layout().get_buffer_size().count(), 25);

    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], float(output_ptr[i]));
    }
}

TEST(pooling_forward_gpu, fs_b_yx_fsv32_avg_65x5x6x7_input_3x3_pool_4x4_stride_3x2_outpad_2x3_inpad)
{
    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    const int features = 65;
    const int batches = 5;
    const int x_input = 6;
    const int y_input = 7;

    const int pool_size = 3;
    const int stride_size = 4;
    const int x_out_pad = 3;
    const int y_out_pad = 2;
    const int x_in_pad = 2;
    const int y_in_pad = 3;

    const tensor input_tensor(batches, features, x_input, y_input);

    std::vector<FLOAT16> input_data(batches*features*x_input*y_input);
    for (size_t i = 0; i < input_data.size(); i++)
    {
        input_data[i] = FLOAT16((float)i/float(input_data.size()));
    }

    auto input_prim = engine.allocate_memory({ data_types::f16,format::bfyx,input_tensor });
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> fsv32_results;

    { //GOLDEN TOPOLOGY
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout().with_padding(padding{ {0,0,x_in_pad,y_in_pad},0 })));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::average, { 1,1,pool_size,pool_size }, { 1,1,stride_size,stride_size }, { 0,0,0,0 }, padding{ { 0,0,x_out_pad,y_out_pad },0 }));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<FLOAT16> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    { //FSV32 TOPOLOGY
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ {0,0,x_in_pad, y_in_pad}, 0 })));
        golden_topology.add(pooling("fsv32_pooling", "reorder_input", pooling_mode::average, { 1,1,pool_size,pool_size }, { 1,1,stride_size,stride_size }, { 0,0,0,0 }, padding{ { 0,0,x_out_pad,y_out_pad },0 }));
        golden_topology.add(reorder("reorder_pooling", "fsv32_pooling", layout(data_types::f16, format::bfyx, input_tensor, padding{ { 0,0,x_out_pad,y_out_pad },0 })));

        network fsv32_network(engine, golden_topology);
        fsv32_network.set_input_data("input", input_prim);

        auto outputs = fsv32_network.execute();
        cldnn::mem_lock<FLOAT16> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            fsv32_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(fsv32_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        EXPECT_NEAR(golden_results[i], fsv32_results[i], 0.001f);
    }
}

template <typename InputT, pooling_mode Mode>
class pooling_test_base {
public:
    using output_t = typename pooling_mode_output<InputT, Mode>::type;

    virtual topology build_topology(const engine& /*eng*/) {
        auto input_size = tensor(batch(batch_num()), feature(input_features()), spatial(input_x(), input_y(), input_z()));
        auto input_lay = layout(input_type(),
                                input_format(),
                                input_size);

        topology topo;
        topo.add(input_layout("input", input_lay));
        if (global_pooling())
            topo.add(pooling("pool", "input", pool_mode()));
        else
            topo.add(pooling("pool",
                             "input",
                             pool_mode(),
                             tensor(batch(0), feature(0), spatial(pool_x(), pool_y(), pool_z())),
                             tensor(batch(0), feature(0), spatial(stride_x(), stride_y(), stride_z())),
                             tensor(batch(0), feature(0), spatial(offset_x(), offset_y(), offset_z()))));
        return topo;
    }

    virtual primitive_id output_id() {
        return "pool";
    }

    virtual void run_expect(const VVVVVF<output_t>& expected) {

        auto& eng = get_test_engine();
        auto topo = build_topology(eng);
        auto opts = build_options(
            build_option::optimize_data(true)
        );
        auto net = network(eng, topo, opts);

        auto input_size = tensor(batch(batch_num()), feature(input_features()), spatial(input_x(), input_y(), input_z()));
        auto input_lay = layout(input_type(),
                                input_format(),
                                input_size);
        auto input_mem = eng.allocate_memory(input_lay);
        std::vector<InputT> input_flat(input_lay.get_linear_size(), static_cast<InputT>(0));
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_features(); ++fi)
                for (size_t zi = 0; zi < input_z(); ++zi)
                    for (size_t yi = 0; yi < input_y(); ++yi)
                        for (size_t xi = 0; xi < input_x(); ++xi) {
                            tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            size_t offset = input_lay.get_linear_offset(coords);
                            input_flat[offset] = _input[bi][fi][zi][yi][xi];
                        }
        set_values(input_mem, input_flat);

        net.set_input_data("input", input_mem);
        auto result = net.execute();
        auto out_mem = result.at(output_id()).get_memory();
        auto out_lay = out_mem->get_layout();
        cldnn::mem_lock<output_t> out_ptr(out_mem, get_test_stream());

        std::string kernel;
        for (auto i : net.get_primitives_info()) {
            if (i.original_id == "pool") {
                kernel = i.kernel_id;
            }
        }
        std::cout << kernel << std::endl;
        SCOPED_TRACE("\nkernel: " + kernel);

        ASSERT_EQ(out_lay.data_type, output_type());
        ASSERT_EQ(out_lay.size.batch[0], expected.size());
        ASSERT_EQ(out_lay.size.feature[0], expected[0].size());
        ASSERT_EQ(out_lay.size.spatial[2], expected[0][0].size());
        ASSERT_EQ(out_lay.size.spatial[1], expected[0][0][0].size());
        ASSERT_EQ(out_lay.size.spatial[0], expected[0][0][0][0].size());

        bool compare_with_tolerance = input_type() == data_types::f16;

        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < expected[0].size(); ++fi)
                for (size_t zi = 0; zi < expected[0][0].size(); ++zi)
                    for (size_t yi = 0; yi < expected[0][0][0].size(); ++yi)
                        for (size_t xi = 0; xi < expected[0][0][0][0].size(); ++xi) {
                            tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            size_t offset = out_lay.get_linear_offset(coords);
                            auto ref_val = static_cast<float>(expected[bi][fi][zi][yi][xi]);
                            auto actual_val = static_cast<float>(out_ptr[offset]);
                            if (compare_with_tolerance) {
                                auto tolerance = 1;
                                ASSERT_NEAR(ref_val, actual_val, tolerance)
                                    << "at b= " << bi << ", f= " << fi << ", z= " << zi << ", y= " << yi << ", x= " << xi;
                            } else {
                                EXPECT_TRUE(are_equal(ref_val, actual_val))
                                    << "at b= " << bi << ", f= " << fi << ", z= " << zi << ", y= " << yi << ", x= " << xi;
                            }
                        }
    }

    size_t batch_num() { return _input.size(); }
    size_t input_features() { return _input[0].size(); }
    size_t input_x() { return _input[0][0][0][0].size(); }
    size_t input_y() { return _input[0][0][0].size(); }
    size_t input_z() { return _input[0][0].size(); }

    format::type input_format() { return _input_fmt; }
    data_types input_type() {
        return type_to_data_type<InputT>::value;
    }

    data_types output_type() {
        return type_to_data_type<output_t>::value;
    }

    pooling_mode pool_mode() { return Mode; }
    size_t pool_x() { return _pool_x; }
    size_t pool_y() { return _pool_y; }
    size_t pool_z() { return _pool_z; }
    int stride_x() { return _stride_x; }
    int stride_y() { return _stride_y; }
    int stride_z() { return _stride_z; }
    int offset_x() { return _offset_x; }
    int offset_y() { return _offset_y; }
    int offset_z() { return _offset_z; }
    bool global_pooling() { return _global_pooling; }

    void set_input(format::type input_fmt, VVVVVF<InputT> input_data) {
        _input_fmt = input_fmt;
        _input = std::move(input_data);
    }

    void set_pool_size(size_t x, size_t y, size_t z) {
        _pool_x = x;
        _pool_y = y;
        _pool_z = z;
    }

    void set_strides(int x, int y, int z) {
        _stride_x = x;
        _stride_y = y;
        _stride_z = z;
    }

    void set_offsets(int x, int y, int z) {
        _offset_x = x;
        _offset_y = y;
        _offset_z = z;
    }

    void set_global_pooling(bool global_pooling) {
        _global_pooling = global_pooling;
    }

    VVVVVF<InputT> _input;
    format::type _input_fmt;
    size_t _pool_x, _pool_y, _pool_z;
    int _stride_x, _stride_y, _stride_z;
    int _offset_x, _offset_y, _offset_z;
    bool _global_pooling;
};

using pooling_random_test_params = std::tuple<
    size_t,                             // batch
    size_t,                             // features
    std::tuple<size_t, size_t, size_t>, // input x, y, z
    std::tuple<size_t, size_t, size_t>, // pool x, y, z
    std::tuple<int, int, int>,          // stride x, y, z
    std::tuple<int, int, int>,          // offset x, y, z
    format::type,                       // input format
    bool                                // global pooling
>;

template <typename InputT, pooling_mode Mode>
class pooling_random_test_base : public pooling_test_base<InputT, Mode> {
public:
    using parent = pooling_test_base<InputT, Mode>;
    using output_t = typename parent::output_t;

    virtual VVVVVF<output_t> calculate_reference() {
        VVVVVF<output_t> reference(this->batch_num(), VVVVF<output_t>(this->input_features()));
        for (size_t bi = 0; bi < this->batch_num(); ++bi) {
            for (size_t fi = 0; fi < this->input_features(); ++fi) {
                reference[bi][fi] = reference_pooling<InputT, Mode>(
                    this->_input[bi][fi],
                    this->pool_x(),
                    this->pool_y(),
                    this->pool_z(),
                    this->stride_x(),
                    this->stride_y(),
                    this->stride_z(),
                    this->offset_x(),
                    this->offset_y(),
                    this->offset_z(),
                    this->global_pooling());
            }
        }
        return reference;
    }

    virtual void param_set_up(const pooling_random_test_params& params) {
        size_t b, f, in_x, in_y, in_z, p_x, p_y, p_z;
        int s_x, s_y, s_z, o_x, o_y, o_z;
        format::type in_fmt;
        bool global_pooling;

        std::forward_as_tuple(
            b,
            f,
            std::forward_as_tuple(in_x, in_y, in_z),
            std::forward_as_tuple(p_x, p_y, p_z),
            std::forward_as_tuple(s_x, s_y, s_z),
            std::forward_as_tuple(o_x, o_y, o_z),
            in_fmt,
            global_pooling
        ) = params;

        auto input_data = generate_random_5d<InputT>(b, f, in_z, in_y, in_x, -256, 256);

        this->set_input(in_fmt, std::move(input_data));
        if (global_pooling) {
            this->set_pool_size(0, 0, 0);
            this->set_strides(1, 1, 1);
            this->set_offsets(0, 0, 0);
        } else {
            this->set_pool_size(p_x, p_y, p_z);
            this->set_strides(s_x, s_y, s_z);
            this->set_offsets(o_x, o_y, o_z);
        }
        this->set_global_pooling(global_pooling);
    }

    void run_random(const pooling_random_test_params& params) {
        param_set_up(params);
        auto reference = calculate_reference();
        ASSERT_NO_FATAL_FAILURE(this->run_expect(reference));
    }
};

using max_pooling_i8_random_test = pooling_random_test_base<int8_t, pooling_mode::max>;
using max_pooling_u8_random_test = pooling_random_test_base<uint8_t, pooling_mode::max>;
using avg_pooling_i8_random_test = pooling_random_test_base<int8_t, pooling_mode::average>;
using avg_pooling_u8_random_test = pooling_random_test_base<uint8_t, pooling_mode::average>;

struct pooling_random_test : public testing::TestWithParam<pooling_random_test_params> {};

TEST_P(pooling_random_test, max_i8) {
    auto test_case = max_pooling_i8_random_test();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test, max_u8) {
    auto test_case = max_pooling_u8_random_test();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test, avg_i8) {
    auto test_case = avg_pooling_i8_random_test();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test, avg_u8) {
    auto test_case = avg_pooling_u8_random_test();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_low_precision_2d_spatial,
    pooling_random_test,
    testing::Combine(testing::Values(1, 2),
                     testing::Values(3, 8, 64),
                     testing::Values(std::tuple<size_t, size_t, size_t>(12, 12, 1), std::tuple<size_t, size_t, size_t>(24, 24, 1)),
                     testing::Values(std::tuple<size_t, size_t, size_t>(4, 4, 1), std::tuple<size_t, size_t, size_t>(2, 2, 1)),
                     testing::Values(std::tuple<int, int, int>(2, 2, 1)),
                     testing::Values(std::tuple<int, int, int>(0, 0, 0)),
                     testing::Values(format::yxfb,
                                     format::bfyx,
                                     format::b_fs_yx_fsv4,
                                     format::b_fs_yx_fsv16,
                                     format::b_fs_yx_fsv32),
                     testing::Values(false, true)),
                    testing::internal::DefaultParamName<pooling_random_test_params>);

INSTANTIATE_TEST_SUITE_P(
    smoke_low_precision_3d_spatial,
    pooling_random_test,
    testing::Combine(testing::Values(1, 2),
                     testing::Values(3, 8, 64),
                     testing::Values(std::tuple<size_t, size_t, size_t>(12, 12, 12), std::tuple<size_t, size_t, size_t>(24, 24, 24)),
                     testing::Values(std::tuple<size_t, size_t, size_t>(4, 4, 4), std::tuple<size_t, size_t, size_t>(2, 2, 2)),
                     testing::Values(std::tuple<int, int, int>(2, 2, 2)),
                     testing::Values(std::tuple<int, int, int>(0, 0, 0)),
                     testing::Values(format::bfzyx,
                                     format::b_fs_zyx_fsv16),
                     testing::Values(false, true)),
                    testing::internal::DefaultParamName<pooling_random_test_params>);

INSTANTIATE_TEST_SUITE_P(
    batched_low_precision,
    pooling_random_test,
    testing::Combine(
        testing::Values(16),
        testing::Values(16, 32),
        testing::Values(std::tuple<size_t, size_t, size_t>(3, 3, 1), std::tuple<size_t, size_t, size_t>(8, 8, 1)),
        testing::Values(std::tuple<size_t, size_t, size_t>(1, 1, 1), std::tuple<size_t, size_t, size_t>(3, 3, 1)),
        testing::Values(std::tuple<int, int, int>(1, 1, 1)),
        testing::Values(std::tuple<int, int, int>(0, 0, 0)),
        testing::Values(format::bs_fs_yx_bsv16_fsv16),
        testing::Values(false, true)
    ),
    testing::internal::DefaultParamName<pooling_random_test_params>);

template <typename InputT, pooling_mode Mode>
class pooling_scale_random_test_base : public pooling_random_test_base<InputT, Mode> {
public:
    using parent = pooling_random_test_base<InputT, Mode>;
    using output_t = typename parent::output_t;

    topology build_topology(engine& eng) override {
        topology topo = parent::build_topology(eng);

        auto scale_lay = layout(this->output_type(), format::bfyx, tensor(batch(1), feature(this->input_features()), spatial(1, 1, 1, 1)));
        auto scale_mem = eng.allocate_memory(scale_lay);
        auto shift_mem = eng.allocate_memory(scale_lay);
        set_values(scale_mem, _scale);
        set_values(shift_mem, _shift);

        topo.add(data("scale_scale", scale_mem));
        topo.add(data("scale_shift", shift_mem));
        topo.add(scale("scale", parent::output_id(), "scale_scale", "scale_shift"));
        topo.add(reorder("scale_wa_out", "scale", this->input_format(), this->output_type()));

        return topo;
    }

    primitive_id output_id() override {
        return "scale_wa_out";
    }

    VVVVVF<output_t> calculate_reference() override {
        auto expected = parent::calculate_reference();

        for (size_t bi = 0; bi < this->batch_num(); ++bi)
            for (size_t fi = 0; fi < this->input_features(); ++fi) {
                expected[bi][fi] = reference_scale_post_op<output_t>(expected[bi][fi], _scale[fi], _shift[fi]);
            }
        return expected;
    }

    void param_set_up(const pooling_random_test_params& params) override {
        parent::param_set_up(params);
        _scale = generate_random_1d<output_t>(this->input_features(), -1, 1);
        _shift = generate_random_1d<output_t>(this->input_features(), -32, 32);
    }

private:
    VF<output_t> _scale;
    VF<output_t> _shift;
};

using pooling_random_test_fp16_fp32 = pooling_random_test;

TEST_P(pooling_random_test_fp16_fp32, avg_fp16) {
    auto test_case = pooling_random_test_base<FLOAT16, pooling_mode::average>();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test_fp16_fp32, max_fp16) {
    auto test_case = pooling_random_test_base<FLOAT16, pooling_mode::max>();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test_fp16_fp32, avg_fp32) {
    auto test_case = pooling_random_test_base<float, pooling_mode::average>();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

TEST_P(pooling_random_test_fp16_fp32, max_fp32) {
    auto test_case = pooling_random_test_base<float, pooling_mode::max>();
    ASSERT_NO_FATAL_FAILURE(test_case.run_random(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_low_precision,
    pooling_random_test_fp16_fp32,
    testing::Combine(testing::Values(1, 2),
                     testing::Values(3, 8),
                     testing::Values(std::tuple<size_t, size_t, size_t>(12, 12, 1), std::tuple<size_t, size_t, size_t>(24, 24, 1)),
                     testing::Values(std::tuple<size_t, size_t, size_t>(4, 4, 1), std::tuple<size_t, size_t, size_t>(2, 2, 1)),
                     testing::Values(std::tuple<int, int, int>(2, 2, 1)),
                     testing::Values(std::tuple<int, int, int>(0, 0, 0)),
                     testing::Values(format::yxfb,
                                     format::bfyx,
                                     format::byxf,
                                     format::b_fs_yx_fsv16,
                                     format::fs_b_yx_fsv32,
                                     format::b_fs_yx_fsv32,
                                     format::b_fs_yx_fsv4),
                     testing::Values(false)),
    testing::internal::DefaultParamName<pooling_random_test_params>);

TEST(pooling_forward_gpu, bsv16_fsv16_max_16x16x8x8_input_2x2_pool_2x2_stride)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 16;
    const int x_input = 8;
    const int y_input = 8;

    const int pool_size = 2;
    const int stride_size = 2;
    const int x_in_pad = 2;
    const int y_in_pad = 2;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({data_types::f32, format::bfyx, input_tensor});
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bfzyx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling",
                                    layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }


    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_max_16x16x2x2_input_4x4_pool_1x1_stride_1x1_inpad)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 16;
    const int x_input = 2;
    const int y_input = 2;

    const int pool_size = 4;
    const int stride_size = 1;
    const int x_in_pad = 1;
    const int y_in_pad = 1;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({data_types::f32, format::bfyx, input_tensor});
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(
                pooling("golden_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                        {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++) {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(
                pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                        {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling", layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_avg_16x16x20x20_input_5x5_pool_3x3_stride)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 16;
    const int x_input = 20;
    const int y_input = 20;

    const int pool_size = 5;
    const int stride_size = 3;
    const int x_in_pad = 0;
    const int y_in_pad = 0;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({data_types::f32, format::bfyx, input_tensor});
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::average, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::average, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size, stride_size}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling",
                                    layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_avg_16x16x20x20_input_5x5_pool_3x1_stride)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 16;
    const int x_input = 20;
    const int y_input = 20;

    const int pool_size = 5;
    const int stride_size_y = 3;
    const int stride_size_x = 1;
    const int x_in_pad = 0;
    const int y_in_pad = 0;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({data_types::f32, format::bfyx, input_tensor});
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::average, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input", layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::average, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling", layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_max_16x16x20x20_input_5x5_pool_3x1_stride)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 16;
    const int x_input = 20;
    const int y_input = 20;

    const int pool_size = 5;
    const int stride_size_y = 3;
    const int stride_size_x = 1;
    const int x_in_pad = 0;
    const int y_in_pad = 0;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,input_tensor });
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(
                pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                        {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling", layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_max_32x32x20x20_input_5x5_pool_3x1_stride)
{
    auto& engine = get_test_engine();

    const int features = 32;
    const int batches = 32;
    const int x_input = 20;
    const int y_input = 20;

    const int pool_size = 5;
    const int stride_size_y = 3;
    const int stride_size_x = 1;
    const int x_in_pad = 0;
    const int y_in_pad = 0;

    const tensor input_tensor(batches, features, x_input, y_input);

    auto input_data = generate_random_1d<float>(batches * features * x_input * y_input, -10, 10);

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,input_tensor });
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(
                pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                        {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling", layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

TEST(pooling_forward_gpu, bsv16_fsv16_max_32x16x20x20_input_5x5_pool_3x1_stride)
{
    auto& engine = get_test_engine();

    const int features = 16;
    const int batches = 32;
    const int x_input = 20;
    const int y_input = 20;

    const int pool_size = 5;
    const int stride_size_y = 3;
    const int stride_size_x = 1;
    const int x_in_pad = 0;
    const int y_in_pad = 0;

    const tensor input_tensor(batches, features, x_input, y_input);

    std::vector<float> input_data(batches * features * x_input * y_input);
    for (size_t i = 0; i < input_data.size(); i++)
    {
        input_data[i] = static_cast<float>(i);
    }

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,input_tensor });
    set_values(input_prim, input_data);

    std::vector<float> golden_results;
    std::vector<float> bsv16_fsv16_results;

    {
        //  golden topology
        topology golden_topology;
        golden_topology.add(input_layout("input", input_prim->get_layout()));
        golden_topology.add(reorder("reorder_input", "input", input_prim->get_layout()));
        golden_topology.add(pooling("golden_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                                    {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));

        network golden_network(engine, golden_topology);
        golden_network.set_input_data("input", input_prim);

        auto outputs = golden_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.begin()->second.get_memory(), get_test_stream());

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            golden_results.push_back(float(output_ptr[i]));
        }
    }

    {
        //  bs_fs_yx_bsv16_fsv16 topology
        topology tested_topology;
        tested_topology.add(input_layout("input", input_prim->get_layout()));
        tested_topology.add(reorder("reorder_input", "input",
                                    layout(data_types::f32, format::bs_fs_yx_bsv16_fsv16, input_tensor)));
        tested_topology.add(
                pooling("bsv16_fsv16_pooling", "reorder_input", pooling_mode::max, {1, 1, pool_size, pool_size},
                        {1, 1, stride_size_x, stride_size_y}, {0, 0, -x_in_pad, -y_in_pad}));
        tested_topology.add(reorder("reorder_pooling", "bsv16_fsv16_pooling", layout(data_types::f32, format::bfyx, input_tensor)));

        build_options op;
        op.set_option(build_option::outputs({"bsv16_fsv16_pooling", "reorder_pooling"}));
        network bsv16_fsv16_network(engine, tested_topology, op);
        bsv16_fsv16_network.set_input_data("input", input_prim);

        auto outputs = bsv16_fsv16_network.execute();
        cldnn::mem_lock<float> output_ptr(outputs.at("reorder_pooling").get_memory(), get_test_stream());

        ASSERT_EQ(outputs.at("bsv16_fsv16_pooling").get_memory()->get_layout().format, format::bs_fs_yx_bsv16_fsv16);

        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            bsv16_fsv16_results.push_back(float(output_ptr[i]));
        }
    }

    ASSERT_EQ(bsv16_fsv16_results.size(), golden_results.size());
    for (size_t i = 0; i < golden_results.size(); i++)
    {
        auto equal = are_equal(golden_results[i], bsv16_fsv16_results[i]);
        EXPECT_TRUE(equal);
        if (!equal)
        {
            std::cout << "Difference at idx = " << i << std::endl;
            return;
        }
    }
}

class pooling_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        all_generic_params.clear();
        all_layer_params.clear();
    }

    static tensor generate_input_offset(int x, int y, const tensor& window_size)
    {
        return tensor(0, 0, -std::min(x, window_size.spatial[0] - 1), -std::min(y, window_size.spatial[1] - 1));
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params()
    {
        std::vector<pooling_mode> pooling_modes = { pooling_mode::max, pooling_mode::average, pooling_mode::average_no_padding };

        std::vector<tensor> sizes = { tensor(1, 1, 2, 2 ), tensor(1, 1, 3, 3), tensor(1, 1, 7, 4) };

        std::vector<tensor> strides = { tensor(1, 1, 1, 1), tensor(1, 1, 2, 2), tensor(1, 1, 4, 3) };

        for (auto pooling_mode : pooling_modes)
        {
            for (auto size : sizes)
            {
                for (auto stride : strides)
                {
                    // No padding
                    all_layer_params.emplace_back(new pooling("pooling", "input0", pooling_mode, size, stride));
                    all_layer_params.emplace_back(new pooling("pooling", "input0", pooling_mode, size, stride, generate_input_offset(4, 3, size)));

                    // Input padding
                    all_layer_params.emplace_back(new pooling("pooling", "reorder0", pooling_mode, size, stride));

                    // Output padding
                    all_layer_params.emplace_back(new pooling("pooling", "input0", pooling_mode, size, stride, generate_input_offset(2, 3, size), { { 0, 0, 1, 5 },{ 0, 0, 19, 4 } }));

                    // Input + output padding
                    all_layer_params.emplace_back(new pooling("pooling", "reorder0", pooling_mode, size, stride, generate_input_offset(2, 3, size), { { 0, 0, 2, 1 },{ 0, 0, 3, 4 } }));
                }
            }
        }

        // This case tests the pooling_gpu_bfyx_average_opt kernel.
        all_layer_params.emplace_back(new pooling("pooling", "input0", pooling_mode::average, tensor(1, 1, 3, 3), tensor(1, 1, 1, 1), generate_input_offset(1, 1, tensor(1, 1, 3, 3))));

        return all_layer_params;
    }

    static std::vector<std::shared_ptr<tests::test_params>> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    bool is_format_supported(cldnn::format format) override
    {
        if ((format == cldnn::format::yxfb) || (format == cldnn::format::bfyx) || (format == cldnn::format::bfyx))
        {
            return true;
        }
        return false;
    }

    void prepare_input_for_test(std::vector<cldnn::memory::ptr>& inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            prepare_input_for_test_typed<float>(inputs);
        }
        else
        {
            prepare_input_for_test_typed<FLOAT16>(inputs);
        }
    }

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory::ptr>& inputs)
    {
        int k = (generic_params->data_type == data_types::f32) ? 8 : 4;
        auto input = inputs[0];
        auto input_size = inputs[0]->get_layout().size;
        VVVVF<Type> input_rnd = generate_random_4d<Type>(input_size.batch[0], input_size.feature[0], input_size.spatial[1], input_size.spatial[0], -10, 10, k);
        VF<Type> input_rnd_vec = flatten_4d<Type>(input->get_layout().format, input_rnd);
        set_values(input, input_rnd_vec);
    }

    cldnn::tensor get_expected_output_tensor() override
    {
        auto pooling = std::static_pointer_cast<cldnn::pooling>(layer_params);

        int batch = generic_params->input_layouts[0].size.batch[0];
        int feature = generic_params->input_layouts[0].size.feature[0];
        int height = generic_params->input_layouts[0].size.spatial[1];
        int width = generic_params->input_layouts[0].size.spatial[0];

        int input_offset_height = pooling->input_offset.spatial[1];
        int input_offset_width = pooling->input_offset.spatial[0];

        int kernel_height = pooling->size.spatial[1];
        int kernel_width = pooling->size.spatial[0];

        int stride_height = pooling->stride.spatial[1];
        int stride_width = pooling->stride.spatial[0];

        int pooled_height = (int)(ceil((float)std::max(height - 2 * input_offset_height - kernel_height, 0) / stride_height)) + 1;
        int pooled_width = (int)(ceil((float)std::max(width - 2 * input_offset_width - kernel_width, 0) / stride_width)) + 1;

        // Make sure that the last pooling starts strictly inside the image.
        while ((pooled_height - 1) * stride_height >= height - input_offset_height)
        {
            --pooled_height;
        }
        while ((pooled_width - 1) * stride_width >= width - input_offset_width)
        {
            --pooled_width;
        }

        return cldnn::tensor(batch, feature, pooled_width, pooled_height);
    }

    template<typename Type>
    memory::ptr generate_reference_typed(const std::vector<cldnn::memory::ptr>& inputs) {
        auto pooling = std::static_pointer_cast<cldnn::pooling>(layer_params);

        int batch = inputs[0]->get_layout().size.batch[0];
        int feature = inputs[0]->get_layout().size.feature[0];
        int height = inputs[0]->get_layout().size.spatial[1];
        int width = inputs[0]->get_layout().size.spatial[0];

        cldnn::pooling_mode pooling_mode = pooling->mode;

        int input_offset_width = pooling->input_offset.spatial[0];
        int input_offset_height = pooling->input_offset.spatial[1];

        int kernel_width = pooling->size.spatial[0];
        int kernel_height = pooling->size.spatial[1];

        int stride_width = pooling->stride.spatial[0];
        int stride_height = pooling->stride.spatial[1];

        auto output_tensor = get_expected_output_tensor();

        int pooled_width = output_tensor.spatial[0];
        int pooled_height = output_tensor.spatial[1];

        //Output is bfyx
        auto output = engine.allocate_memory(cldnn::layout(inputs[0]->get_layout().data_type, cldnn::format::bfyx, output_tensor, pooling->output_padding));

        cldnn::mem_lock<Type> input_mem(inputs[0], get_test_stream());
        cldnn::mem_lock<Type> output_mem(output, get_test_stream());

        int output_width = output->get_layout().get_buffer_size().spatial[0];
        int output_height = output->get_layout().get_buffer_size().spatial[1];

        const auto input_desc = get_linear_memory_desc(inputs[0]->get_layout());
        const auto output_desc = get_linear_memory_desc(output->get_layout());

        switch (pooling_mode)
        {
            case cldnn::pooling_mode::max:
            {
                for (int i = 0; i < (int)output->get_layout().get_buffer_size().count(); i++)
                {
                    output_mem[i] = (generic_params->data_type == data_types::f32) ? -FLT_MAX : -65504;
                }
                for (int b = 0; b < batch; b++)
                {
                    for (int f = 0; f < feature; f++)
                    {
                        for (int h = 0; h < pooled_height; h++)
                        {
                            for (int w = 0; w < pooled_width; w++)
                            {
                                int input_offset_x_start = w * stride_width + input_offset_width;
                                int input_offset_x_end = std::min(input_offset_x_start + kernel_width, width);
                                input_offset_x_start = std::max(input_offset_x_start, 0);

                                int input_offset_y_start = h * stride_height + input_offset_height;
                                int input_offset_y_end = std::min(input_offset_y_start + kernel_height, height);
                                input_offset_y_start = std::max(input_offset_y_start, 0);

                                const size_t output_index = get_linear_index(output->get_layout(), b, f, h, w, output_desc);

                                for (int y = input_offset_y_start; y < input_offset_y_end; y++)
                                {
                                    for (int x = input_offset_x_start; x < input_offset_x_end; x++)
                                    {
                                        const size_t input_index = get_linear_index(inputs[0]->get_layout(), b, f, y, x, input_desc);

                                        if (input_mem[input_index] > output_mem[output_index])
                                        {
                                            output_mem[output_index] = input_mem[input_index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case cldnn::pooling_mode::average:
            case cldnn::pooling_mode::average_no_padding:
            {
                auto dynamic_mode = (((output_tensor.spatial[0] - 1) * stride_width) + pooling->size.spatial[0]) > -2 * input_offset_width + width ||
                    (((output_tensor.spatial[1] - 1) * stride_height) + pooling->size.spatial[1]) > -2 * input_offset_width + height;

                auto divider = [=](int actual_x, int actual_y) {
                    auto x = kernel_width;
                    auto y = kernel_height;
                    if (dynamic_mode)
                    {
                        if (actual_x + kernel_width > width + std::abs(input_offset_width))
                        {
                            x = (width + std::abs(input_offset_width)) - actual_x;
                        }
                        if (actual_y + kernel_height > height + std::abs(input_offset_height))
                        {
                            y = (height + std::abs(input_offset_height)) - actual_y;
                        }
                    }
                    return y*x;
                };

                for (int i = 0; i < (int)output->get_layout().get_buffer_size().count(); i++)
                {
                    output_mem[i] = 0;
                }
                for (int b = 0; b < batch; b++)
                {
                    for (int f = 0; f < feature; f++)
                    {
                        for (int h = 0; h < pooled_height; h++)
                        {
                            for (int w = 0; w < pooled_width; w++)
                            {
                                int input_offset_x_start = w * stride_width + input_offset_width;
                                int input_offset_x_end = std::min(input_offset_x_start + kernel_width, width);
                                input_offset_x_start = std::max(input_offset_x_start, 0);

                                int input_offset_y_start = h * stride_height + input_offset_height;
                                int input_offset_y_end = std::min(input_offset_y_start + kernel_height, height);
                                input_offset_y_start = std::max(input_offset_y_start, 0);

                                int output_index = (b * feature + f) * output_height * output_width;
                                tensor lower_padding = pooling->output_padding.lower_size();
                                output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

                                int num_of_elements = 0;
                                for (int y = input_offset_y_start; y < input_offset_y_end; y++)
                                {
                                    for (int x = input_offset_x_start; x < input_offset_x_end; x++)
                                    {
                                        const size_t input_index = get_linear_index(inputs[0]->get_layout(), b, f, y, x, input_desc);
                                        output_mem[output_index] += input_mem[input_index];
                                        if (!dynamic_mode || pooling_mode == cldnn::pooling_mode::average_no_padding)
                                        {
                                            num_of_elements++;
                                        }
                                    }
                                }
                                if (pooling_mode == cldnn::pooling_mode::average)
                                {
                                        num_of_elements = divider(input_offset_x_start, input_offset_y_start);
                                }
                                if (num_of_elements == 0)
                                {
                                    assert(0);
                                    return output;
                                }
                                output_mem[output_index] /= (Type)num_of_elements;

                            }
                        }
                    }
                }
                break;
            }
            default:
            {
                assert(0);
            }
        }

        return output;
    }

    memory::ptr generate_reference(const std::vector<cldnn::memory::ptr>& inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

private:

    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;

};

std::vector<std::shared_ptr<cldnn::primitive>> pooling_test::all_layer_params = {};
std::vector<std::shared_ptr<tests::test_params>> pooling_test::all_generic_params = {};

TEST_P(pooling_test, POOLING)
{
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_POOLING,
                        pooling_test,
                        ::testing::Combine(::testing::ValuesIn(pooling_test::generate_generic_test_params()),
                                           ::testing::ValuesIn(pooling_test::generate_specific_test_params())),
                        tests::generic_test::custom_param_name_functor());
