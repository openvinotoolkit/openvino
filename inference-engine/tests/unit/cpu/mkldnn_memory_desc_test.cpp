// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>

#include "mkldnn_memory.h"
#include "details/ie_exception.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

TEST(MemDescTest, Conversion) {
    // Check if conversion keep desc structure
    // dnnl::memory::desc -> MKLDNNMemoryDesc -> TensorDesc -> MKLDNNMemoryDesc -> dnnl::memory::desc
    auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
        dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
        MKLDNNMemoryDesc plg_tdesc {orig_tdesc};
        TensorDesc ie_tdesc {plg_tdesc};
        MKLDNNMemoryDesc plg_tdesc_after {ie_tdesc};
        dnnl::memory::desc after_tdesc(plg_tdesc_after);

        return  orig_tdesc == after_tdesc;
    };

    std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
        { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
        { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
        { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
        { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
        { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } }   // blocked and permuted outer dims
    };

    for (const auto &p : payload)
        ASSERT_TRUE(converted_correctly(p.first, p.second));
}

TEST(MemDescTest, ConversionKeepAny) {
    dnnl::memory::desc tdesc {{1, 2, 3, 4}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::any};
    MKLDNNMemoryDesc plg_tdesc {tdesc};
    TensorDesc ie_tdesc {plg_tdesc};
    MKLDNNMemoryDesc plg_tdesc_2 {ie_tdesc};
    dnnl::memory::desc tdesc_2 {plg_tdesc_2};

    ASSERT_TRUE(tdesc == tdesc_2);
}

TEST(MemDescTest, isPlainCheck) {
    const auto dims = dnnl::memory::dims {3, 2, 5, 7};
    const auto type = dnnl::memory::data_type::u8;
    dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
    dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::acdb};
    dnnl::memory::desc blckd_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};

    ASSERT_TRUE(MKLDNNMemoryDesc(plain_tdesc).isPlainFormat());
    ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).isPlainFormat());
    ASSERT_FALSE(MKLDNNMemoryDesc(blckd_tdesc).isPlainFormat());
}

TEST(MemDescTest, CheckScalar) {
    SKIP();
}

TEST(MemDescTest, UpperBound) {
    SKIP();
}

TEST(MemDescTest, BlockedConversion) {
    SKIP();
}

TEST(MemDescTest, ComaptibleWithFormat) {
    SKIP();
}
