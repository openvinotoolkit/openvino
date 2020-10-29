// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "../../../thirdparty/mkl-dnn/src/common/memory_desc_wrapper.hpp"
#undef UNUSED
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct mkldnn2TD_test_params {
    std::vector<size_t> dims;

    mkldnn::memory::format mkldnnFormat;
};

class MKLDNN2TensorDescConvertTests: public TestsCommon,
                                     public WithParamInterface<mkldnn2TD_test_params> {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            mkldnn2TD_test_params p = ::testing::WithParamInterface<mkldnn2TD_test_params>::GetParam();

            mkldnn::memory::dims mkldnnDims;
            for (auto dim : p.dims) {
                mkldnnDims.push_back(dim);
            }
            MKLDNNPlugin::MKLDNNMemoryDesc mkldnnMemoryDesc(mkldnnDims, mkldnn::memory::data_type::f32, p.mkldnnFormat);
            mkldnn::memory::desc mkldnnDesc = mkldnnMemoryDesc;
            mkldnn::impl::memory_desc_wrapper dst_d(mkldnnDesc.data);
            InferenceEngine::TensorDesc tDesc = mkldnnMemoryDesc;

            size_t total_size = std::accumulate(std::begin(p.dims), std::end(p.dims), (size_t) 1, std::multiplies<size_t>());

            for (size_t i = 0; i < total_size; i++) {
                ASSERT_EQ(tDesc.offset(i), dst_d.off_l(i));
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNN2TensorDescConvertTests, TestsConvertation) {}


INSTANTIATE_TEST_CASE_P(
        TestsConvertation, MKLDNN2TensorDescConvertTests,
        ::testing::Values(
                mkldnn2TD_test_params{{5}, mkldnn::memory::format::x},
                mkldnn2TD_test_params{{10, 3}, mkldnn::memory::format::nc},
                mkldnn2TD_test_params{{1, 3, 8, 8}, mkldnn::memory::format::nhwc},
                mkldnn2TD_test_params{{1, 3, 8, 8}, mkldnn::memory::format::nchw},
                mkldnn2TD_test_params{{1, 32, 8, 8}, mkldnn::memory::format::nChw8c},
                mkldnn2TD_test_params{{1, 8, 8, 8}, mkldnn::memory::format::nChw8c},
                mkldnn2TD_test_params{{1, 32, 8, 8}, mkldnn::memory::format::nChw16c},
                mkldnn2TD_test_params{{67, 34}, mkldnn::memory::format::oi},
                mkldnn2TD_test_params{{1, 3, 8, 8}, mkldnn::memory::format::oihw},
                mkldnn2TD_test_params{{1, 3, 8, 8}, mkldnn::memory::format::nChw8c},
                mkldnn2TD_test_params{{1, 32, 8, 8}, mkldnn::memory::format::nChw16c},
                mkldnn2TD_test_params{{1, 16, 8, 8}, mkldnn::memory::format::oIhw8i}
        ));

struct TD2mkldnn_test_params {
    std::vector<size_t> dims;
    std::vector<size_t> blocked_dims;
    std::vector<size_t> order;
};

class TensorDesc2MKLDNNConvertTests: public TestsCommon,
                                     public WithParamInterface<TD2mkldnn_test_params> {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            TD2mkldnn_test_params p = ::testing::WithParamInterface<TD2mkldnn_test_params>::GetParam();

            mkldnn::memory::dims mkldnnDims;
            for (auto dim : p.dims) {
                mkldnnDims.push_back(dim);
            }
            InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, p.dims, {p.blocked_dims, p.order});
            MKLDNNPlugin::MKLDNNMemoryDesc desc(tDesc);

            mkldnn::impl::memory_desc_wrapper dst_d(((mkldnn::memory::desc&)desc).data);

            size_t total_size = std::accumulate(std::begin(p.dims), std::end(p.dims), (size_t) 1, std::multiplies<size_t>());

            for (size_t i = 0; i < total_size; i++) {
                ASSERT_EQ(tDesc.offset(i), dst_d.off_l(i));
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(TensorDesc2MKLDNNConvertTests, TestsConvertation) {}


INSTANTIATE_TEST_CASE_P(
        TestsConvertation, TensorDesc2MKLDNNConvertTests,
        ::testing::Values(
                TD2mkldnn_test_params{{5}, {5}, {0}},
                TD2mkldnn_test_params{{10, 3}, {10, 3}, {0, 1}},
                TD2mkldnn_test_params{{1, 3, 8, 8}, {1, 8, 8, 3}, {0, 2, 3, 1}},
                TD2mkldnn_test_params{{1, 3, 8, 8}, {1, 3, 8, 8}, {0, 1, 2, 3}},
                TD2mkldnn_test_params{{1, 8, 8, 8}, {1, 1, 8, 8, 8}, {0, 1, 2, 3, 1}},
                TD2mkldnn_test_params{{1, 32, 8, 8}, {1, 2, 8, 8, 16}, {0, 1, 2, 3, 1}},
                TD2mkldnn_test_params{{1, 3, 8}, {1, 3, 8}, {0, 1, 2}}
//                TD2mkldnn_test_params{{1, 3, 8, 8}, mkldnn::memory::format::nchw},
//                TD2mkldnn_test_params{{1, 3, 8, 8}, mkldnn::memory::format::nChw8c},
//                TD2mkldnn_test_params{{1, 32, 8, 8}, mkldnn::memory::format::nChw16c},
//                TD2mkldnn_test_params{{67, 34}, mkldnn::memory::format::oi},
//                TD2mkldnn_test_params{{1, 3, 8, 8}, mkldnn::memory::format::oihw},
//                TD2mkldnn_test_params{{5, 1, 3, 8, 8}, mkldnn::memory::format::goihw},
//                TD2mkldnn_test_params{{1, 16, 8, 8}, mkldnn::memory::format::oIhw8i},
//                TD2mkldnn_test_params{{1, 3, 8, 8}, mkldnn::memory::format::OhIw16o4i}
        ));
