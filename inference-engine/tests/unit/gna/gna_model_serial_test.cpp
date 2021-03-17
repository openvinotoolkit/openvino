// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna_model_serial.hpp"

using ::testing::Return;
using ::testing::_;

class IstreamMock final: public std::streambuf {
public:
    MOCK_METHOD3(seekoff, std::streampos(std::streamoff, std::ios_base::seekdir,
                            std::ios_base::openmode));
};

TEST(GNAModelSerialTest, TestErrorOnTellg) {
    IstreamMock mock;
    EXPECT_CALL(mock, seekoff(_, _, _)).WillRepeatedly(Return(-1));
    std::istream is(&mock);
    ASSERT_THROW(GNAModelSerial::ReadHeader(is), InferenceEngine::details::InferenceEngineException);
}
