// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"

using namespace cldnn;
using namespace ::tests;

TEST(serialization_gpu, layout_from_string) {
    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);

        std::stringstream ss;
        ss << InferenceEngine::Layout::ANY;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::NCHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::NHWC;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::NCDHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::NDHWC;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::OIHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::GOIHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::OIDHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::GOIDHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::SCALAR;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::C;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::CHW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::HWC;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::HW;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::NC;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::CN;
        ob << ss.str();

        ss.str("");
        ss << InferenceEngine::Layout::BLOCKED;
        ob << ss.str();
    }
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib = BinaryInputBuffer(in_mem, get_test_engine());
    
        std::string layout;
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::ANY);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::NCHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::NHWC);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::NCDHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::NDHWC);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::OIHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::GOIHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::OIDHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::GOIDHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::SCALAR);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::C);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::CHW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::HWC);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::HW);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::NC);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::CN);
        ib >> layout;
        ASSERT_EQ(cldnn::layout_from_string(layout), InferenceEngine::Layout::BLOCKED);
    }
}
