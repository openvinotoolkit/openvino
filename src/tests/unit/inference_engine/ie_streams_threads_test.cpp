// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <vector>
#include "threading/ie_istreams_executor.hpp"
#include <fstream>

using DefaultParams = std::tuple<int,  // num_cores
                                int,  // num_cores_phy
                                int,  // num_big_cores_phy
                                int,  // stream_mode
                                int,  // big_core_streams
                                int,  // small_core_streams
                                int,  // threads_per_stream_big
                                int   // threads_per_stream_small
                                >;
using CustomStreamsParams = std::tuple<int,  // num_cores
                                      int,  // num_cores_phy
                                      int,  // num_big_cores_phy
                                      int,  // streams number that custom set
                                      int,  // threads number that custom set
                                      int,  // big_core_streams
                                      int,  // small_core_streams
                                      int,  // threads_per_stream_big
                                      int   // threads_per_stream_small
                                      >;
class DefaultStreamsTest : public ::testing::TestWithParam<DefaultParams>{};
class CustomStreamsSetTest : public ::testing::TestWithParam<CustomStreamsParams>{};

const std::vector<DefaultParams> defaultConfigs = {
    // 1P+4E, no hyper thread, ADL:Intel® Celeron® 7300/7305/7305E/7305L
    DefaultParams{5, 5, 1, 0, 1, 4, 1, 1},  // mode_type=0
    DefaultParams{5, 5, 1, 1, 1, 4, 1, 1},  // mode_type=1
    DefaultParams{5, 5, 1, 2, 1, 2, 1, 2},  // mode_type=2
    // 1p+4E, ADL:Intel® Pentium® Gold 8500 /8505
    DefaultParams{6, 5, 1, 0, 2, 4, 1, 1},  // mode_type=0
    DefaultParams{6, 5, 1, 1, 2, 4, 1, 1},  // mode_type=1
    DefaultParams{6, 5, 1, 2, 1, 2, 2, 2},  // mode_type=2
    // 2P+0E, ADL:Intel® Pentium® Gold G7400/G7400E/G7400T/G7400TE
    DefaultParams{4, 2, 2, 0, 2, 0, 2, 0},
    DefaultParams{4, 2, 2, 1, 4, 0, 1, 0},
    DefaultParams{4, 2, 2, 2, 2, 0, 2, 0},
    // 2P+4E, ADL:Intel® Core™ i3-1210U/i3-1215U/i3-1215UE/i3-1215UL
    DefaultParams{8, 6, 2, 0, 2, 2, 2, 2},
    DefaultParams{8, 6, 2, 1, 4, 4, 1, 1},
    DefaultParams{8, 6, 2, 2, 2, 2, 2, 2},
    // 2P+8E, ADL:Intel® Core™
    // i3-1220P/i5-1230U/i5-1235U/i5-1235UL/i5-1240U/i5-1245U/i5-1245UE/i5-1245UL/i7-1250U/i7-1255U/i7-1255UL/i7-1260U/i7-1265U/i7-1265UE/i7-1265UL
    DefaultParams{12, 10, 2, 0, 2, 4, 2, 2},
    DefaultParams{12, 10, 2, 1, 4, 8, 1, 1},
    DefaultParams{12, 10, 2, 2, 2, 4, 2, 2},
    // 4P+0E, ADL:Intel® Core™ i3-12100/i3-12100E/i3-12100F/i3-12100T/i3-12100TE/i3-12300/i3-12300T
    DefaultParams{8, 4, 4, 0, 2, 0, 4, 0},
    DefaultParams{8, 4, 4, 1, 8, 0, 1, 0},
    DefaultParams{8, 4, 4, 2, 4, 0, 2, 0},
    // 4P+4E, ADL:Intel® Core™ i3-1220PE/i3-12300HE/i3-12300HL/i5-12450H/i5-12450HX
    DefaultParams{12, 8, 4, 0, 2, 1, 4, 4},
    DefaultParams{12, 8, 4, 1, 8, 4, 1, 1},
    DefaultParams{12, 8, 4, 2, 4, 2, 2, 2},
    // 4P+8E, ADL:Intel® Core™
    // i5-1240P/i5-1250P/i5-1250PE/i5-12500H/i5-12500HL/i5-12600H/i5-12600HE/i5-12600HL/i5-12600HX/i7-1260P/i7-1270P/i7-1270PE
    DefaultParams{16, 12, 4, 0, 2, 2, 4, 4},
    DefaultParams{16, 12, 4, 1, 8, 8, 1, 1},
    DefaultParams{16, 12, 4, 2, 4, 4, 2, 2},
    // 6P+0E, ADL:Intel® Core™ i5-12400/i5-12400F/i5-12400T/i5-12500/i5-12500E/i5-12500T/i5-12500TE/i5-12600/i5-12600T
    DefaultParams{12, 6, 6, 0, 4, 0, 3, 0},
    DefaultParams{12, 6, 6, 1, 12, 0, 1, 0},
    DefaultParams{12, 6, 6, 2, 6, 0, 2, 0},
    // 6P+4E, ADL:Intel® Core™ i5-12600K/i5-12600KF/i7-12650H
    DefaultParams{16, 10, 6, 0, 4, 1, 3, 3},
    DefaultParams{16, 10, 6, 1, 12, 4, 1, 1},
    DefaultParams{16, 10, 6, 2, 6, 2, 2, 2},
    // 6P+8E, ADL:Intel® Core™
    // i5-13600K/i5-13600KF/i7-12650HX/i7-12700H/i7-12700HL/i7-1280P/i7-12800H/i7-12800HE/i7-12800HL/i9-12900H/i9-12900HK,
    // RPL:i5-13600K/i5-13600KF
    DefaultParams{20, 14, 6, 0, 4, 2, 3, 3},
    DefaultParams{20, 14, 6, 1, 12, 8, 1, 1},
    DefaultParams{20, 14, 6, 2, 6, 4, 2, 2},
    // 8P+4E, ADL:Intel® Core™ i7-12700/i7-12700E /i7-12700F/ i7-12700T/ i7-12700TE/i7-12700K/ i7-12700KF
    DefaultParams{20, 12, 8, 0, 4, 1, 4, 4},
    DefaultParams{20, 12, 8, 1, 16, 4, 1, 1},
    DefaultParams{20, 12, 8, 2, 8, 2, 2, 2},
    // 8P+8E, ADL:Intel® Core™
    // i7-12800HX/i7-12850HX/i7-13700K/i7-13700KF/i9-12900/i9-12900E/i9-12900F/i9-12900T/i9-12900TE/i9-12900K/i9-12900KF/i9-12900KS/i9-12900HX/i9-12950HX
    // RPL:i7-13700K/i7-13700KF
    DefaultParams{24, 16, 8, 0, 4, 2, 4, 4},
    DefaultParams{24, 16, 8, 1, 16, 8, 1, 1},
    DefaultParams{24, 16, 8, 2, 8, 4, 2, 2},
    // 8P+16E, ADL/RPL:Intel® Core™ i9-13900K/i9-13900KF
    DefaultParams{32, 24, 8, 0, 4, 4, 4, 4},
    DefaultParams{32, 24, 8, 1, 16, 16, 1, 1},
    DefaultParams{32, 24, 8, 2, 8, 8, 2, 2}};

const std::vector<CustomStreamsParams> customConfigs = {
    CustomStreamsParams{5, 5, 1, 5, 0, 1, 4, 1, 1},       // 1P+4E, nstreams=5, nthreads=0
    CustomStreamsParams{5, 5, 1, 3, 0, 1, 2, 1, 1},       // 1P+4E, nstreams=3, nthreads=0
    CustomStreamsParams{5, 5, 1, 2, 5, 0, 2, 0, 2},       // 1P+4E, nstreams=2, nthreads=5
    CustomStreamsParams{5, 5, 1, 3, 6, 1, 2, 1, 1},       // 1P+4E, nstreams=3, nthreads=6
    CustomStreamsParams{6, 5, 1, 6, 0, 2, 4, 1, 1},       // 1P+4E, nstreams=6, nthreads=0
    CustomStreamsParams{6, 5, 1, 4, 0, 1, 3, 1, 1},       // 1P+4E, nstreams=4, nthreads=0
    CustomStreamsParams{6, 5, 1, 2, 4, 0, 2, 0, 2},       // 1P+4E, nstreams=2, nthreads=4
    CustomStreamsParams{6, 5, 1, 3, 6, 1, 2, 1, 1},       // 1P+4E, nstreams=3, nthreads=6
    CustomStreamsParams{4, 2, 2, 4, 0, 4, 0, 1, 0},       // 2P+0E, nstreams=4, nthreads=0
    CustomStreamsParams{4, 2, 2, 2, 0, 2, 0, 1, 0},       // 2P+0E, nstreams=2, nthreads=0
    CustomStreamsParams{4, 2, 2, 2, 3, 2, 0, 1, 0},       // 2P+0E, nstreams=2, nthreads=3
    CustomStreamsParams{4, 2, 2, 4, 6, 4, 0, 1, 0},       // 2P+0E, nstreams=4, nthreads=6
    CustomStreamsParams{8, 6, 2, 8, 0, 4, 4, 1, 1},       // 2P+4E, nstreams=8, nthreads=0
    CustomStreamsParams{8, 6, 2, 4, 0, 2, 2, 2, 2},       // 2P+4E, nstreams=4, nthreads=0
    CustomStreamsParams{8, 6, 2, 2, 0, 1, 1, 2, 2},       // 2P+4E, nstreams=2, nthreads=0
    CustomStreamsParams{8, 6, 2, 2, 4, 1, 1, 2, 2},       // 2P+4E, nstreams=2, nthreads=4
    CustomStreamsParams{8, 6, 2, 3, 8, 1, 2, 2, 2},       // 2P+4E, nstreams=3, nthreads=8
    CustomStreamsParams{12, 10, 2, 12, 0, 4, 8, 1, 1},    // 2P+8E, nstreams=12, nthreads=0
    CustomStreamsParams{12, 10, 2, 6, 0, 2, 4, 2, 2},     // 2P+8E, nstreams=6, nthreads=0
    CustomStreamsParams{12, 10, 2, 4, 0, 1, 3, 2, 2},     // 2P+8E, nstreams=4, nthreads=0
    CustomStreamsParams{12, 10, 2, 6, 12, 2, 4, 2, 2},    // 2P+8E, nstreams=6, nthreads=12
    CustomStreamsParams{12, 10, 2, 3, 9, 1, 2, 2, 2},     // 2P+8E, nstreams=3, nthreads=9
    CustomStreamsParams{8, 4, 4, 8, 0, 8, 0, 1, 0},       // 4P+0E, nstreams=8, nthreads=0
    CustomStreamsParams{8, 4, 4, 4, 0, 4, 0, 1, 0},       // 4P+0E, nstreams=4, nthreads=0
    CustomStreamsParams{8, 4, 4, 2, 0, 2, 0, 1, 0},       // 4P+0E, nstreams=2, nthreads=0
    CustomStreamsParams{8, 4, 4, 4, 8, 4, 0, 1, 0},       // 4P+0E, nstreams=4, nthreads=8
    CustomStreamsParams{8, 4, 4, 3, 9, 3, 0, 1, 0},       // 4P+0E, nstreams=3, nthreads=9
    CustomStreamsParams{12, 8, 4, 12, 0, 8, 4, 1, 1},     // 4P+4E, nstreams=12, nthreads=0
    CustomStreamsParams{12, 8, 4, 8, 0, 4, 4, 1, 1},      // 4P+4E, nstreams=8, nthreads=0
    CustomStreamsParams{12, 8, 4, 3, 0, 2, 1, 4, 4},      // 4P+4E, nstreams=3, nthreads=0
    CustomStreamsParams{12, 8, 4, 6, 12, 4, 2, 2, 2},     // 4P+4E, nstreams=6, nthreads=12
    CustomStreamsParams{12, 8, 4, 2, 14, 1, 1, 4, 4},     // 4P+4E, nstreams=2, nthreads=14
    CustomStreamsParams{16, 12, 4, 16, 0, 8, 8, 1, 1},    // 4P+8E, nstreams=16, nthreads=0
    CustomStreamsParams{16, 12, 4, 8, 0, 4, 4, 2, 2},     // 4P+8E, nstreams=8, nthreads=0
    CustomStreamsParams{16, 12, 4, 4, 0, 2, 2, 4, 4},     // 4P+8E, nstreams=4, nthreads=0
    CustomStreamsParams{16, 12, 4, 4, 16, 2, 2, 4, 4},    // 4P+8E, nstreams=4, nthreads=16
    CustomStreamsParams{16, 12, 4, 7, 20, 3, 4, 2, 2},    // 4P+8E, nstreams=7, nthreads=20
    CustomStreamsParams{12, 6, 6, 12, 0, 12, 0, 1, 0},    // 6P+0E, nstreams=12, nthreads=0
    CustomStreamsParams{12, 6, 6, 6, 0, 6, 0, 1, 0},      // 6P+0E, nstreams=6, nthreads=0
    CustomStreamsParams{12, 6, 6, 4, 0, 4, 0, 1, 0},      // 6P+0E, nstreams=4, nthreads=0
    CustomStreamsParams{12, 6, 6, 6, 16, 6, 0, 1, 0},     // 6P+0E, nstreams=6, nthreads=16
    CustomStreamsParams{12, 6, 6, 5, 14, 5, 0, 1, 0},     // 6P+0E, nstreams=5, nthreads=14
    CustomStreamsParams{16, 10, 6, 16, 0, 12, 4, 1, 1},   // 6P+4E, nstreams=16, nthreads=0
    CustomStreamsParams{16, 10, 6, 8, 0, 6, 2, 2, 2},     // 6P+4E, nstreams=8, nthreads=0
    CustomStreamsParams{16, 10, 6, 5, 0, 4, 1, 3, 3},     // 6P+4E, nstreams=5, nthreads=0
    CustomStreamsParams{16, 10, 6, 8, 16, 6, 2, 2, 2},    // 6P+4E, nstreams=8, nthreads=16
    CustomStreamsParams{16, 10, 6, 3, 13, 2, 1, 4, 4},    // 6P+4E, nstreams=3, nthreads=13
    CustomStreamsParams{20, 14, 6, 20, 0, 12, 8, 1, 1},   // 6P+8E, nstreams=20, nthreads=0
    CustomStreamsParams{20, 14, 6, 10, 0, 6, 4, 2, 2},    // 6P+8E, nstreams=10, nthreads=0
    CustomStreamsParams{20, 14, 6, 6, 0, 4, 2, 3, 3},     // 6P+8E, nstreams=6, nthreads=0
    CustomStreamsParams{20, 14, 6, 8, 20, 4, 4, 2, 2},    // 6P+8E, nstreams=8, nthreads=20
    CustomStreamsParams{20, 14, 6, 4, 16, 2, 2, 4, 4},    // 6P+8E, nstreams=4, nthreads=16
    CustomStreamsParams{20, 12, 8, 20, 0, 16, 4, 1, 1},   // 8P+4E, nstreams=20, nthreads=0
    CustomStreamsParams{20, 12, 8, 16, 0, 12, 4, 1, 1},   // 8P+4E, nstreams=16, nthreads=0
    CustomStreamsParams{20, 12, 8, 5, 0, 4, 1, 4, 4},     // 8P+4E, nstreams=5, nthreads=0
    CustomStreamsParams{20, 12, 8, 16, 20, 12, 4, 1, 1},  // 8P+4E, nstreams=16, nthreads=20
    CustomStreamsParams{20, 12, 8, 4, 12, 3, 1, 3, 3},    // 8P+4E, nstreams=4, nthreads=12
    CustomStreamsParams{24, 16, 8, 24, 0, 16, 8, 1, 1},   // 8P+8E, nstreams=24, nthreads=0
    CustomStreamsParams{24, 16, 8, 16, 0, 8, 8, 1, 1},    // 8P+8E, nstreams=16, nthreads=0
    CustomStreamsParams{24, 16, 8, 6, 0, 4, 2, 4, 4},     // 8P+8E, nstreams=6, nthreads=0
    CustomStreamsParams{24, 16, 8, 16, 24, 8, 8, 1, 1},   // 8P+8E, nstreams=16, nthreads=24
    CustomStreamsParams{24, 16, 8, 9, 20, 5, 4, 2, 2},    // 8P+8E, nstreams=9, nthreads=20
    CustomStreamsParams{32, 24, 8, 32, 0, 16, 16, 1, 1},  // 8P+16E, nstreams=32, nthreads=0
    CustomStreamsParams{32, 24, 8, 16, 0, 8, 8, 2, 2},    // 8P+16E, nstreams=16, nthreads=0
    CustomStreamsParams{32, 24, 8, 8, 0, 4, 4, 4, 4},     // 8P+16E, nstreams=8, nthreads=0
    CustomStreamsParams{32, 24, 8, 18, 30, 8, 10, 1, 1},  // 8P+16E, nstreams=18, nthreads=30
    CustomStreamsParams{32, 24, 8, 25, 28, 9, 16, 1, 1}   // 8P+16E, nstreams=25, nthreads=28
};

TEST_P(DefaultStreamsTest, getHybridNumStreams) {
    int num_cores = 0;
    int num_cores_phy = 0;
    int num_big_cores_phy = 0;
    int stream_mode = 0;
    int big_core_streams = 0;
    int small_core_streams = 0;
    int threads_per_stream_big = 0;
    int threads_per_stream_small = 0;
    std::tie(num_cores,
             num_cores_phy,
             num_big_cores_phy,
             stream_mode,
             big_core_streams,
             small_core_streams,
             threads_per_stream_big,
             threads_per_stream_small) = GetParam();
    int streams = big_core_streams + small_core_streams;
    auto executorConfig = InferenceEngine::IStreamsExecutor::Config();
    std::map<std::string, std::string> config = {};
    InferenceEngine::IStreamsExecutor::Config::CpuInfo cpu_info = {num_cores, num_cores_phy, num_big_cores_phy};
    EXPECT_EQ(executorConfig.GetHybridNumStreams(config, stream_mode, &cpu_info), streams);

    EXPECT_EQ(big_core_streams, std::stoi(config["BIG_CORE_STREAMS"]));
    EXPECT_EQ(small_core_streams, std::stoi(config["SMALL_CORE_STREAMS"]));
    EXPECT_EQ(threads_per_stream_big, std::stoi(config["THREADS_PER_STREAM_BIG"]));
    EXPECT_EQ(threads_per_stream_small, std::stoi(config["THREADS_PER_STREAM_SMALL"]));
}

TEST_P(CustomStreamsSetTest, updateHybridCustomThreads) {
    int num_cores = 0;
    int num_cores_phy = 0;
    int num_big_cores_phy = 0;
    int custom_nstreams = 0;
    int custom_nthreads = 0;
    int big_core_streams = 0;
    int small_core_streams = 0;
    int threads_per_stream_big = 0;
    int threads_per_stream_small = 0;
    std::tie(num_cores,
             num_cores_phy,
             num_big_cores_phy,
             custom_nstreams,
             custom_nthreads,
             big_core_streams,
             small_core_streams,
             threads_per_stream_big,
             threads_per_stream_small) = GetParam();
    auto executorConfig = InferenceEngine::IStreamsExecutor::Config();
    InferenceEngine::IStreamsExecutor::Config::CpuInfo cpu_info = {num_cores, num_cores_phy, num_big_cores_phy};
    executorConfig._streams = custom_nstreams;
    executorConfig._threads = custom_nthreads;
    executorConfig.UpdateHybridCustomThreads(executorConfig, &cpu_info);

    EXPECT_EQ(big_core_streams, executorConfig._big_core_streams);
    EXPECT_EQ(small_core_streams, executorConfig._small_core_streams);
    EXPECT_EQ(threads_per_stream_big, executorConfig._threads_per_stream_big);
    EXPECT_EQ(threads_per_stream_small, executorConfig._threads_per_stream_small);
}

INSTANTIATE_TEST_SUITE_P(smoke_default_hybrid_num_streams, DefaultStreamsTest,
                ::testing::ValuesIn(defaultConfigs));

INSTANTIATE_TEST_SUITE_P(smoke_custom_hybrid_num_streams, CustomStreamsSetTest,
                ::testing::ValuesIn(customConfigs));