// TODO [mandrono]: uncomment

// // Copyright (C) 2018-2021 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include <utility>
// #include <gtest/gtest.h>

// #include "mkldnn_memory.h"

// using namespace MKLDNNPlugin;
// using namespace InferenceEngine;

// TEST(MemDescTest, Conversion) {
//     // Check if conversion keep desc structure
//     // dnnl::memory::desc -> MKLDNNMemoryDesc -> TensorDesc -> MKLDNNMemoryDesc -> dnnl::memory::desc
//     auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
//         dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
//         MKLDNNMemoryDesc plg_tdesc {orig_tdesc};
//         TensorDesc ie_tdesc {plg_tdesc};
//         MKLDNNMemoryDesc plg_tdesc_after {ie_tdesc};
//         dnnl::memory::desc after_tdesc(plg_tdesc_after);

//         return  orig_tdesc == after_tdesc;
//     };

//     std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
//         { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
//         { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
//         { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
//         { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
//         { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } },  // blocked and permuted outer dims
//         { dnnl::memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
//     };

//     for (const auto &p : payload)
//         ASSERT_TRUE(converted_correctly(p.first, p.second));
// }

// TEST(MemDescTest, CompareWithTensorDescRecomputedStrides) {
//     auto converted_correctly = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
//         dnnl::memory::desc orig_tdesc {dims, dnnl::memory::data_type::u8, fmt};
//         MKLDNNMemoryDesc plg_tdesc {orig_tdesc};
//         TensorDesc ie_tdesc {plg_tdesc};

//         const BlockingDesc block_dess(ie_tdesc.getBlockingDesc().getBlockDims(), ie_tdesc.getBlockingDesc().getOrder());
//         TensorDesc recomputed_tdesc(ie_tdesc.getPrecision(), ie_tdesc.getDims(), block_dess);

//         return  ie_tdesc == recomputed_tdesc;
//     };

//     std::pair<dnnl::memory::format_tag, dnnl::memory::dims> payload[] {
//         { dnnl::memory::format_tag::nChw16c,     {1, 1, 10, 10} },  // auto blocked
//         { dnnl::memory::format_tag::nhwc,        {4, 2, 10, 7 } },  // permuted
//         { dnnl::memory::format_tag::nchw,        {4, 2, 10, 7 } },  // plain
//         { dnnl::memory::format_tag::NChw16n16c,  {4, 2, 10, 7 } },  // blocked for 2 dims
//         { dnnl::memory::format_tag::BAcd16a16b,  {4, 2, 10, 7 } },  // blocked and permuted outer dims
//         { dnnl::memory::format_tag::Acdb16a,     {96, 1, 7, 7 } },  // same strides but not default order
//     };

//     for (const auto &p : payload)
//         ASSERT_TRUE(converted_correctly(p.first, p.second));
// }

// TEST(MemDescTest, ConversionKeepAny) {
//     dnnl::memory::desc tdesc {{1, 2, 3, 4}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::any};
//     MKLDNNMemoryDesc plg_tdesc {tdesc};
//     TensorDesc ie_tdesc {plg_tdesc};
//     MKLDNNMemoryDesc plg_tdesc_2 {ie_tdesc};
//     dnnl::memory::desc tdesc_2 {plg_tdesc_2};

//     ASSERT_TRUE(tdesc == tdesc_2);
// }

// TEST(MemDescTest, isPlainCheck) {
//     const auto dims = dnnl::memory::dims {3, 2, 5, 7};
//     const auto type = dnnl::memory::data_type::u8;
//     dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
//     dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::acdb};
//     dnnl::memory::desc blckd_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};

//     ASSERT_TRUE(MKLDNNMemoryDesc(plain_tdesc).isPlainFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).isPlainFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(blckd_tdesc).isPlainFormat());
// }

// TEST(MemDescTest, isBlockedCCheck) {
//     const auto dims = dnnl::memory::dims {3, 2, 5, 7};
//     const auto type = dnnl::memory::data_type::u8;

//     dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
//     dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
//     dnnl::memory::desc blck4_tdesc {dims, type, dnnl::memory::format_tag::aBcd4b};
//     dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
//     dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
//     ASSERT_FALSE(MKLDNNMemoryDesc(plain_tdesc).isBlockedCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(tailc_tdesc).isBlockedCFormat());
//     ASSERT_TRUE(MKLDNNMemoryDesc(blck4_tdesc).isBlockedCFormat());
//     ASSERT_TRUE(MKLDNNMemoryDesc(blck8_tdesc).isBlockedCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).isBlockedCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(blck4_tdesc).isBlockedCFormat(8));
//     ASSERT_TRUE(MKLDNNMemoryDesc(blck4_tdesc).isBlockedCFormat(4));

//     const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
//     const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
//     dnnl::memory::desc blck8_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
//     dnnl::memory::desc blck8_permCD_crop_tdesc = blck8_permCD_tdesc.submemory_desc(crop_dims, crop_off);
//     ASSERT_TRUE(MKLDNNMemoryDesc(blck8_crop_tdesc).isBlockedCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_crop_tdesc).isBlockedCFormat());
// }

// TEST(MemDescTest, isTailCCheck) {
//     const auto dims = dnnl::memory::dims {3, 2, 5, 7};
//     const auto type = dnnl::memory::data_type::u8;

//     dnnl::memory::desc plain_tdesc {dims, type, dnnl::memory::format_tag::abcd};
//     dnnl::memory::desc tailc_tdesc {dims, type, dnnl::memory::format_tag::acdb};
//     dnnl::memory::desc permt_tdesc {dims, type, dnnl::memory::format_tag::bcda};
//     dnnl::memory::desc blck8_tdesc {dims, type, dnnl::memory::format_tag::aBcd8b};
//     ASSERT_FALSE(MKLDNNMemoryDesc(plain_tdesc).isTailCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(permt_tdesc).isTailCFormat());
//     ASSERT_TRUE(MKLDNNMemoryDesc(tailc_tdesc).isTailCFormat());
//     ASSERT_FALSE(MKLDNNMemoryDesc(blck8_tdesc).isTailCFormat());

//     dnnl::memory::desc blck8_permCD_tdesc {dims, type, dnnl::memory::format_tag::aBdc16b};
//     ASSERT_FALSE(MKLDNNMemoryDesc(blck8_permCD_tdesc).isTailCFormat());

//     const auto crop_dims = dnnl::memory::dims {2, 1, 5, 7};
//     const auto crop_off = dnnl::memory::dims {1, 0, 0, 0};
//     dnnl::memory::desc tailc_crop_tdesc = blck8_tdesc.submemory_desc(crop_dims, crop_off);
//     ASSERT_FALSE(MKLDNNMemoryDesc(tailc_crop_tdesc).isTailCFormat());
// }

// TEST(MemDescTest, constructWithPlainFormat) {
//     GTEST_SKIP();
// }

// TEST(MemDescTest, CheckScalar) {
//     GTEST_SKIP();
// }

// TEST(MemDescTest, UpperBound) {
//     GTEST_SKIP();
// }

// TEST(MemDescTest, BlockedConversion) {
//     GTEST_SKIP();
// }

// TEST(MemDescTest, ComaptibleWithFormat) {
//     GTEST_SKIP();
// }

// TEST(isSameMethodTest, CheckTensorWithSameStrides) {
//     auto isSameDataFormat = [] (dnnl::memory::format_tag fmt, dnnl::memory::dims dims) {
//         dnnl::memory::desc oneDnnDesc {dims, dnnl::memory::data_type::u8, fmt};
//         MKLDNNMemoryDesc pluginDesc {oneDnnDesc};
//         return pluginDesc.getFormat() == fmt;
//     };

//     std::pair<dnnl::memory::format_tag, dnnl::memory::dims> testCases[] {
//         { dnnl::memory::format_tag::ntc, {1, 10, 10} },
//     };

//     for (const auto &tc : testCases)
//         ASSERT_TRUE(isSameDataFormat(tc.first, tc.second));
// }
