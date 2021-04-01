// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <mkldnn_memory.h>
#include <ie_common.h>

#include <vector>

#include <common/memory_desc_wrapper.hpp>

namespace IE = InferenceEngine;
using Tag = mkldnn::memory::format_tag;
using RefDesc = mkldnn::impl::memory_desc_wrapper;
using MKLDNNPlugin::MKLDNNMemory;
using MKLDNNPlugin::MKLDNNMemoryDesc;

TEST(TensorDescTests, checkOff) {
    auto workload = std::vector<std::pair<IE::SizeVector, Tag>>{
            {{5},                Tag::a},
            {{10, 3},            Tag::ab},
            {{5,  3},            Tag::ba},
            {{1,  3,  8, 8},     Tag::abcd},
            {{1,  3,  5, 2},     Tag::acdb},
            {{1,  24, 5, 7},     Tag::aBcd8b},
            {{2,  10, 3, 3},     Tag::aBcd8b},
            {{1,  3,  8, 8},     Tag::aBcd8b},
            {{1,  32, 8, 8},     Tag::aBcd16b},
            {{1,  32, 8, 8},     Tag::aBcd16b},
            {{2,  3,  5, 2, 1},  Tag::abcde},
    };

    for (const auto &p : workload) {
            mkldnn::memory::dims dims {p.first.begin(), p.first.end()};

            const auto cpu_tDesc = MKLDNNMemoryDesc {dims, mkldnn::memory::data_type::f32, p.second};
            const auto ie_tDesc  = IE::TensorDesc {cpu_tDesc};

            mkldnn::memory::desc dnnl_tdesc = cpu_tDesc;
            const RefDesc ref(dnnl_tdesc.data);
            size_t total_size = cpu_tDesc.getDims().size();

            for (size_t i = 0; i < total_size; i++) {
                ASSERT_EQ(ie_tDesc.offset(i), ref.off_l(i)) << "Offset calculation are different";
            }
    }
}

TEST(TensorDescTests, convertToFrom) {
    struct Param { IE::SizeVector dims, blk, ord; };
    auto workload = std::vector<Param>{
            {{5}, {5}, {0}},
            {{10, 3}, {10, 3}, {0, 1}},
            {{1, 3, 8, 8}, {1, 8, 8, 3}, {0, 2, 3, 1}},
            {{1, 3, 8, 8}, {1, 3, 8, 8}, {0, 1, 2, 3}},
            {{1, 8, 8, 8}, {1, 1, 8, 8, 8}, {0, 1, 2, 3, 1}},
            {{1, 32, 8, 8}, {1, 2, 8, 8, 16}, {0, 1, 2, 3, 1}},
            {{1, 3, 8}, {1, 3, 8}, {0, 1, 2}}
    };

    for (const auto &p : workload) {
        const auto ie_tDesc  = IE::TensorDesc(IE::Precision::FP32, p.dims, {p.blk, p.ord});
        const auto cpu_tDesc = MKLDNNMemoryDesc {ie_tDesc};

        mkldnn::memory::desc dnnl_tdesc = cpu_tDesc;
        const RefDesc ref(dnnl_tdesc.data);
        size_t total_size = cpu_tDesc.getDims().size();

        for (size_t i = 0; i < total_size; i++) {
            ASSERT_EQ(ie_tDesc.offset(i), ref.off_l(i)) << "Offset calculation are different";
        }
    }
}
