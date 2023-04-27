// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"

using namespace cldnn;
using namespace ::tests;

struct ie_layout_serialization_test : testing::TestWithParam<InferenceEngine::Layout> {
    void run_test() {
        InferenceEngine::Layout test_layout = GetParam();

        membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);

            std::stringstream ss;
            ss << test_layout;
            ob << ss.str();
        }
        {
            std::istream in_mem(&mem_buf);
            BinaryInputBuffer ib = BinaryInputBuffer(in_mem, get_test_engine());
        
            std::string str_layout;
            ib >> str_layout;
            EXPECT_EQ(cldnn::serial_util::layout_from_string(str_layout), test_layout);
        }
    }
};

TEST_P(ie_layout_serialization_test, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    gpu_serialization,
    ie_layout_serialization_test,
    testing::Values(InferenceEngine::Layout::ANY,
                    InferenceEngine::Layout::NCHW,
                    InferenceEngine::Layout::NHWC,
                    InferenceEngine::Layout::NCDHW,
                    InferenceEngine::Layout::NDHWC,
                    InferenceEngine::Layout::OIHW,
                    InferenceEngine::Layout::GOIHW,
                    InferenceEngine::Layout::OIDHW,
                    InferenceEngine::Layout::GOIDHW,
                    InferenceEngine::Layout::SCALAR,
                    InferenceEngine::Layout::C,
                    InferenceEngine::Layout::CHW,
                    InferenceEngine::Layout::HWC,
                    InferenceEngine::Layout::HW,
                    InferenceEngine::Layout::NC,
                    InferenceEngine::Layout::CN,
                    InferenceEngine::Layout::BLOCKED)
);
