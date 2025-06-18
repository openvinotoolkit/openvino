// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log_dispatch.hpp"

#include <gtest/gtest.h>

namespace ov::test {

using BufferCaptureParams = std::pair<std::ostream*, ov::util::LogStream*>;

class BufferCapture : public testing::TestWithParam<BufferCaptureParams> {
protected:
    void SetUp() override {
        std::tie(out_stream, log_stream) = GetParam();
        out_stream->flush();
        log_stream->flush();
        out_buf = out_stream->rdbuf();
        out_stream->rdbuf(str_stream.rdbuf());
    }

    void TearDown() override {
        out_stream->rdbuf(out_buf);
    }

    ov::util::LogStream* log_stream;
    std::ostream* out_stream;
    std::streambuf* out_buf;
    std::stringstream str_stream;
};

TEST_P(BufferCapture, default_insert) {
    *log_stream << "TEST 123" << std::endl;
    EXPECT_EQ(str_stream.str(), "TEST 123\n");

    *out_stream << "test abc" << std::endl;
    EXPECT_EQ(str_stream.str(), "TEST 123\ntest abc\n");
}

INSTANTIATE_TEST_SUITE_P(LogDispatch,
                         BufferCapture,
                         ::testing::ValuesIn(std::vector<BufferCaptureParams>{{&std::cout, &ov_cout},
                                                                              {&std::cerr, &ov_cerr}}));

}  // namespace ov::test
