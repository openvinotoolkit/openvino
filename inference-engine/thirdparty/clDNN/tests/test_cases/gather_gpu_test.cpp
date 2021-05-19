// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/gather.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(gather7_gpu_fp16, d222_axisX_bdim_m1) {
    //  Dictionary : 2x2x2x2x2x2  
    //  Indexes : 2x2x2x1
    //  Axis : 5
    //  batch_dim : -1
    //  Output : 2x2x2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f 1.f 0.f 0.f 0.f 0.f 1.f 0.f
    //
    //  Dictionary:
    //  1.f   2.f   3.f   4.f   5.f   6.f   7.f   8.f   9.f   10.f  11.f  12.f  13.f  14.f  15.f  16.f  17.f  18.f
    //  19.f  20.f  21.f  22.f  23.f  24.f  25.f  26.f  27.f  28.f  29.f  30.f  31.f  32.f  33.f  34.f  35.f  36.f
    //  37.f  38.f  39.f  40.f  41.f  42.f  43.f  44.f  45.f  46.f  47.f  48.f  49.f  50.f  51.f  52.f  53.f  54.f
    //  55.f  56.f  57.f  58.f  59.f  60.f  61.f  62.f  63.f  64.f
    //
    //  Output:
    //  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f, 
    //  9.f,  10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 
    //  17.f, 17.f, 19.f, 19.f, 21.f, 21.f, 23.f, 23.f, 
    //  25.f, 25.f, 27.f, 27.f, 29.f, 29.f, 31.f, 31.f, 
    //  33.f, 33.f, 35.f, 35.f, 37.f, 37.f, 39.f, 39.f, 
    //  41.f, 41.f, 43.f, 43.f, 45.f, 45.f, 47.f, 47.f, 
    //  50.f, 49.f, 52.f, 51.f, 54.f, 53.f, 56.f, 55.f, 
    //  58.f, 57.f, 60.f, 59.f, 62.f, 61.f, 64.f, 63.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2} }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 2 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_x;
    int64_t batch_dim = -1;

    set_values(input1, {
        FLOAT16(1.f),   FLOAT16(2.f),   FLOAT16(3.f),   FLOAT16(4.f),   FLOAT16(5.f),   FLOAT16(6.f),   FLOAT16(7.f),   FLOAT16(8.f),
        FLOAT16(9.f),   FLOAT16(10.f),  FLOAT16(11.f),  FLOAT16(12.f),  FLOAT16(13.f),  FLOAT16(14.f),  FLOAT16(15.f),  FLOAT16(16.f),

        FLOAT16(17.f),  FLOAT16(18.f),  FLOAT16(19.f),  FLOAT16(20.f),  FLOAT16(21.f),  FLOAT16(22.f),  FLOAT16(23.f),  FLOAT16(24.f),
        FLOAT16(25.f),  FLOAT16(26.f),  FLOAT16(27.f),  FLOAT16(28.f),  FLOAT16(29.f),  FLOAT16(30.f),  FLOAT16(31.f),  FLOAT16(32.f),

        FLOAT16(33.f),  FLOAT16(34.f),  FLOAT16(35.f),  FLOAT16(36.f),  FLOAT16(37.f),  FLOAT16(38.f),  FLOAT16(39.f),  FLOAT16(40.f),
        FLOAT16(41.f),  FLOAT16(42.f),  FLOAT16(43.f),  FLOAT16(44.f),  FLOAT16(45.f),  FLOAT16(46.f),  FLOAT16(47.f),  FLOAT16(48.f),

        FLOAT16(49.f),  FLOAT16(50.f),  FLOAT16(51.f),  FLOAT16(52.f),  FLOAT16(53.f),  FLOAT16(54.f),  FLOAT16(55.f),  FLOAT16(56.f),
        FLOAT16(57.f),  FLOAT16(58.f),  FLOAT16(59.f),  FLOAT16(60.f),  FLOAT16(61.f),  FLOAT16(62.f),  FLOAT16(63.f),  FLOAT16(64.f),
    });

    set_values(input2, {
        0.f, 1.f,
        0.f, 0.f,
        
        0.f, 0.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfwzyx, tensor(2, 2, 2, 2, 2, 2), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f, 
        9.f,  10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 
        17.f, 17.f, 19.f, 19.f, 21.f, 21.f, 23.f, 23.f, 
        25.f, 25.f, 27.f, 27.f, 29.f, 29.f, 31.f, 31.f, 
        33.f, 33.f, 35.f, 35.f, 37.f, 37.f, 39.f, 39.f, 
        41.f, 41.f, 43.f, 43.f, 45.f, 45.f, 47.f, 47.f, 
        50.f, 49.f, 52.f, 51.f, 54.f, 53.f, 56.f, 55.f, 
        58.f, 57.f, 60.f, 59.f, 62.f, 61.f, 64.f, 63.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d323_axisY_bdim_m1) {
    //  Dictionary : 3x2x3x4x2  
    //  Indexes : 3x2x3x1
    //  Axis : 3
    //  batch_dim : -1
    //  Output : 3x2x3x3x2
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 0.f 3.f 1.f 0.f 1.f 1.f 1.f 2.f 0.f 3.f 3.f 1.f 0.f 2.f 0.f 1.f
    //
    //  Dictionary:
    //  1.f   2.f   3.f   4.f   5.f   6.f   7.f   8.f   9.f   10.f  11.f  12.f  13.f  14.f  15.f  16.f  17.f  18.f
    //  19.f  20.f  21.f  22.f  23.f  24.f  25.f  26.f  27.f  28.f  29.f  30.f  31.f  32.f  33.f  34.f  35.f  36.f
    //  37.f  38.f  39.f  40.f  41.f  42.f  43.f  44.f  45.f  46.f  47.f  48.f  49.f  50.f  51.f  52.f  53.f  54.f
    //  55.f  56.f  57.f  58.f  59.f  60.f  61.f  62.f  63.f  64.f  65.f  66.f  67.f  68.f  69.f  70.f  71.f  72.f
    //  73.f  74.f  75.f  76.f  77.f  78.f  79.f  80.f  81.f  82.f  83.f  84.f  85.f  86.f  87.f  88.f  89.f  90.f
    //  91.f  92.f  93.f  94.f  95.f  96.f  97.f  98.f  99.f  100.f 101.f 102.f 103.f 104.f 105.f 106.f 107.f 108.f
    //  109.f 110.f 111.f 112.f 113.f 114.f 115.f 116.f 117.f 118.f 119.f 120.f 121.f 122.f 123.f 124.f 125.f 126.f
    //  127.f 128.f 129.f 130.f 131.f 132.f 133.f 134.f 135.f 136.f 137.f 138.f 139.f 140.f 141.f 142.f 143.f 144.f
    //
    //  Output:
    //  1.f   2.f   1.f   2.f   1.f   2.f   9.f   10.f   9.f  10.f   9.f  10.f
    //  17.f  18.f  17.f  18.f  17.f  18.f  31.f  32.f  27.f  28.f  25.f  26.f
    //  39.f  40.f  35.f  6.f   33.f  34.f  47.f  48.f  43.f  44.f  41.f  42.f
    //  51.f  52.f  51.f  52.f  51.f  52.f  59.f  60.f  59.f  60.f  59.f  60.f
    //  67.f  68.f  67.f  68.f  67.f  68.f  77.f  78.f  73.f  74.f  79.f  80.f
    //  85.f  86.f  81.f  82.f  87.f  88.f  93.f  94.f  89.f  90.f  95.f  96.f
    //  103.f 104.f  99.f  100.f 97.f  98.f 111.f 112.f 107.f 108.f 105.f 106.f
    //  119.f 120.f 115.f 116.f 113.f 114.f 125.f 126.f 121.f 122.f 123.f 124.f
    //  133.f 134.f 129.f 130.f 131.f 132.f 141.f 142.f 137.f 138.f 139.f 140.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 3, 2, 2, 4, 3} }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 3 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;
    int64_t batch_dim = -1;

    set_values(input1, {
        FLOAT16(1.f),   FLOAT16(2.f),   FLOAT16(3.f),   FLOAT16(4.f),   FLOAT16(5.f),   FLOAT16(6.f),   FLOAT16(7.f),   FLOAT16(8.f),
        FLOAT16(9.f),   FLOAT16(10.f),  FLOAT16(11.f),  FLOAT16(12.f),  FLOAT16(13.f),  FLOAT16(14.f),  FLOAT16(15.f),  FLOAT16(16.f),
        FLOAT16(17.f),  FLOAT16(18.f),  FLOAT16(19.f),  FLOAT16(20.f),  FLOAT16(21.f),  FLOAT16(22.f),  FLOAT16(23.f),  FLOAT16(24.f),

        FLOAT16(25.f),  FLOAT16(26.f),  FLOAT16(27.f),  FLOAT16(28.f),  FLOAT16(29.f),  FLOAT16(30.f),  FLOAT16(31.f),  FLOAT16(32.f),
        FLOAT16(33.f),  FLOAT16(34.f),  FLOAT16(35.f),  FLOAT16(36.f),  FLOAT16(37.f),  FLOAT16(38.f),  FLOAT16(39.f),  FLOAT16(40.f),
        FLOAT16(41.f),  FLOAT16(42.f),  FLOAT16(43.f),  FLOAT16(44.f),  FLOAT16(45.f),  FLOAT16(46.f),  FLOAT16(47.f),  FLOAT16(48.f),


        FLOAT16(49.f),  FLOAT16(50.f),  FLOAT16(51.f),  FLOAT16(52.f),  FLOAT16(53.f),  FLOAT16(54.f),  FLOAT16(55.f),  FLOAT16(56.f),
        FLOAT16(57.f),  FLOAT16(58.f),  FLOAT16(59.f),  FLOAT16(60.f),  FLOAT16(61.f),  FLOAT16(62.f),  FLOAT16(63.f),  FLOAT16(64.f),
        FLOAT16(65.f),  FLOAT16(66.f),  FLOAT16(67.f),  FLOAT16(68.f),  FLOAT16(69.f),  FLOAT16(70.f),  FLOAT16(71.f),  FLOAT16(72.f),

        FLOAT16(73.f),  FLOAT16(74.f),  FLOAT16(75.f),  FLOAT16(76.f),  FLOAT16(77.f),  FLOAT16(78.f),  FLOAT16(79.f),  FLOAT16(80.f),
        FLOAT16(81.f),  FLOAT16(82.f),  FLOAT16(83.f),  FLOAT16(84.f),  FLOAT16(85.f),  FLOAT16(86.f),  FLOAT16(87.f),  FLOAT16(88.f),
        FLOAT16(89.f),  FLOAT16(90.f),  FLOAT16(91.f),  FLOAT16(92.f),  FLOAT16(93.f),  FLOAT16(94.f),  FLOAT16(95.f),  FLOAT16(96.f),


        FLOAT16(97.f),  FLOAT16(98.f),  FLOAT16(99.f),  FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f),
        FLOAT16(105.f), FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f), FLOAT16(112.f),
        FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f), FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),

        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f), FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f),
        FLOAT16(129.f), FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f), FLOAT16(136.f),
        FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f), FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f)
    });

    set_values(input2, {
        0.f, 0.f, 0.f,
        3.f, 1.f, 0.f,
        
        1.f, 1.f, 1.f,
        2.f, 0.f, 3.f,
        
        3.f, 1.f, 0.f,
        2.f, 0.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfzyx, tensor(3, 2, 2, 3, 3), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f,   2.f,   1.f,   2.f,   1.f,   2.f,
        9.f,   10.f,  9.f,   10.f,  9.f,   10.f,
        17.f,  18.f,  17.f,  18.f,  17.f,  18.f,

        31.f,  32.f,  27.f,  28.f,  25.f,  26.f,
        39.f,  40.f,  35.f,  36.f,  33.f,  34.f,
        47.f,  48.f,  43.f,  44.f,  41.f,  42.f,


        51.f,  52.f,  51.f,  52.f,  51.f,  52.f,
        59.f,  60.f,  59.f,  60.f,  59.f,  60.f,
        67.f,  68.f,  67.f,  68.f,  67.f,  68.f,

        77.f,  78.f,  73.f,  74.f,  79.f,  80.f,
        85.f,  86.f,  81.f,  82.f,  87.f,  88.f,
        93.f,  94.f,  89.f,  90.f,  95.f,  96.f,
        

        103.f, 104.f,  99.f,  100.f, 97.f,  98.f,
        111.f, 112.f, 107.f, 108.f, 105.f, 106.f,
        119.f, 120.f, 115.f, 116.f, 113.f, 114.f,

        125.f, 126.f, 121.f, 122.f, 123.f, 124.f,
        133.f, 134.f, 129.f, 130.f, 131.f, 132.f,
        141.f, 142.f, 137.f, 138.f, 139.f, 140.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d44_axisY_bdim1) {
    //  Dictionary : 4x3x5x1
    //  Indexes : 4x4x1x1
    //  Axis : 2
    //  batch_dim : 1
    //  Output : 4x3x4x1
    //  Input values in fp16

    //  Indexes:
    //  3.f 2.f 3.f 4.f 3.f 2.f 2.f 1.f 1.f 1.f 0.f 4.f 2.f 4.f 3.f 2.f
    //
    //  Dictionary:
    //  84.f  7.f 10.f 69.f 13.f 47.f 75.f  8.f 65.f 28.f  5.f 12.f 56.f 54.f  9.f 31.f 12.f 71.f
    //  55.f  8.f 73.f 16.f 29.f 81.f 81.f 75.f  8.f 74.f 75.f 51.f  7.f 29.f  6.f 72.f 18.f 38.f
    //  54.f 19.f 70.f 16.f 74.f 40.f 72.f 88.f 24.f 14.f 75.f 74.f 82.f 25.f 48.f 13.f 71.f 92.f
    //  9.f 73.f  8.f 80.f 27.f 64.f
    //
    //  Output:
    //  69.f 10.f 69.f 13.f 65.f  8.f 65.f 28.f 54.f 56.f 54.f  9.f 55.f 71.f 71.f 12.f 81.f 29.f
    //  29.f 16.f 75.f 74.f 74.f  8.f 29.f 29.f  7.f 18.f 54.f 54.f 38.f 16.f 40.f 40.f 74.f 24.f
    //  74.f 25.f 82.f 74.f 71.f  9.f 92.f 71.f 80.f 64.f 27.f 80.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 4, 3, 1, 5 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 4, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;
    int64_t batch_dim = 1;

    set_values(input1, {
        FLOAT16(84.f), FLOAT16( 7.f), FLOAT16(10.f), FLOAT16(69.f), FLOAT16(13.f),
        FLOAT16(47.f), FLOAT16(75.f), FLOAT16( 8.f), FLOAT16(65.f), FLOAT16(28.f), 
        FLOAT16( 5.f), FLOAT16(12.f), FLOAT16(56.f), FLOAT16(54.f), FLOAT16( 9.f), 
        
        FLOAT16(31.f), FLOAT16(12.f), FLOAT16(71.f), FLOAT16(55.f), FLOAT16( 8.f), 
        FLOAT16(73.f), FLOAT16(16.f), FLOAT16(29.f), FLOAT16(81.f), FLOAT16(81.f), 
        FLOAT16(75.f), FLOAT16( 8.f), FLOAT16(74.f), FLOAT16(75.f), FLOAT16(51.f), 

        FLOAT16( 7.f), FLOAT16(29.f), FLOAT16( 6.f), FLOAT16(72.f), FLOAT16(18.f), 
        FLOAT16(38.f), FLOAT16(54.f), FLOAT16(19.f), FLOAT16(70.f), FLOAT16(16.f), 
        FLOAT16(74.f), FLOAT16(40.f), FLOAT16(72.f), FLOAT16(88.f), FLOAT16(24.f), 
        
        FLOAT16(14.f), FLOAT16(75.f), FLOAT16(74.f), FLOAT16(82.f), FLOAT16(25.f), 
        FLOAT16(48.f), FLOAT16(13.f), FLOAT16(71.f), FLOAT16(92.f), FLOAT16( 9.f), 
        FLOAT16(73.f), FLOAT16( 8.f), FLOAT16(80.f), FLOAT16(27.f), FLOAT16(64.f)
    });

    set_values(input2, {
        3.f, 2.f, 3.f, 4.f,
        3.f, 2.f, 2.f, 1.f,
        1.f, 1.f, 0.f, 4.f,
        2.f, 4.f, 3.f, 2.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(4, 3, 1, 4), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        69.f, 10.f, 69.f, 13.f, 
        65.f,  8.f, 65.f, 28.f, 
        54.f, 56.f, 54.f,  9.f, 

        55.f, 71.f, 71.f, 12.f, 
        81.f, 29.f, 29.f, 16.f, 
        75.f, 74.f, 74.f,  8.f,

        29.f, 29.f,  7.f, 18.f, 
        54.f, 54.f, 38.f, 16.f, 
        40.f, 40.f, 74.f, 24.f,

        74.f, 25.f, 82.f, 74.f, 
        71.f,  9.f, 92.f, 71.f, 
        80.f, 64.f, 27.f, 80.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim_m1) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : -1
    //  Output : 3x2x1x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;
    size_t batch_dim = -1;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(3, 2, 1, 1), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 1.f, 
        4.f, 3.f,
        5.f, 5.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim1) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : 1
    //  Output : 3x2x1x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;
    int64_t batch_dim = 1;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(3, 2, 1, 1), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 1.f, 4.f,
        3.f, 5.f, 5.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim0) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : 0
    //  Output : 3x3x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;
    size_t batch_dim = 0;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(3, 3, 1, 2), batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 1.f, 
        2.f, 1.f,
        1.f, 1.f,

        3.f, 3.f, 
        4.f, 3.f,
        3.f, 3.f,

        5.f, 5.f, 
        6.f, 5.f,
        5.f, 5.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(3.0f), FLOAT16(4.0f)
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f,
        2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
            FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

            FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
            FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 2, 3, 4, 3, 4, 1, 2

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
            1, 2,
            3, 4
    });

    set_values(input2, {
            0, 1,
            1, 0
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 3, 4, 1, 2
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
            1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d41_axisB) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 4x1x2x3
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //  7, 8, 9, 10, 11, 12
    //  1, 2, 3, 4, 5, 6,

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
               });

    set_values(input2, {
            0, 1, 1, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(4, 1, 3, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, d41_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 2x4x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  1, 0, 1, 2
    //
    //  Dictionary:
    //  1, 2,   3, 4,   5, 6,
    //  7, 8,   9, 10,  11, 12
    //
    //  Output:
    //  3, 4,   1, 2,   3, 4,   5, 6,
    //  9, 10,  7, 8,   9, 10,  11, 12

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f
               });

    set_values(input2, {
            1, 0, 1, 2
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 4, 2, 1))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            9.f, 10.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, d2_axisX) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 2x1x1x1
    //  Axis : 0
    //  Output : 2x2x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 1, 2, 2, 3, 3, 4, 4

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 1, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_x;

    set_values(input1, {
            1.f, 2.f,
            3.f, 4.f,
               });

    set_values(input2, {
            0, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(2, 2, 2, 1))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 1.f, 2.f, 2.f,
            3.f, 3.f, 4.f, 4.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, 322_axisF) {
    //  Dictionary : 3x3x1x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 3x2x2x1
    //  Input values in i32

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 3, 3, 1, 1 } }); // data
    auto input2 = memory::allocate(engine, { data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
        0, 1, 2,  10, 11, 12,   20, 21, 22
    });

    set_values(input2, {
        1, 0,
        2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, format::bfyx, tensor(3, 2, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<int>();

    std::vector<int> expected_results = {
        1, 0, 2, 1,   11, 10, 12, 11,   21, 20, 22, 21
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << i;
    }
}
