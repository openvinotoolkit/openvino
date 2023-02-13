// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "common/versioning.hpp"
#include "gna_model_serial.hpp"

using namespace testing;

class IstreamMock final : public std::streambuf {
public:
    MOCK_METHOD3(seekoff, std::streampos(std::streamoff, std::ios_base::seekdir, std::ios_base::openmode));
};

TEST(GNAModelSerialTest, TestErrorOnTellg) {
    IstreamMock mock;
    EXPECT_CALL(mock, seekoff(_, _, _)).WillRepeatedly(Return(-1));
    std::istream is(&mock);
    ASSERT_THROW(GNAModelSerial::ReadHeader(is), InferenceEngine::Exception);
}

TEST(GNAVersionSerializerTest, Export) {
    std::stringstream sBuf;
    GNAVersionSerializer verSerializer;
    verSerializer.Export(sBuf);

    EXPECT_THAT(sBuf.str(), HasSubstr(ov::intel_gna::common::get_openvino_version_string()));
}

TEST(GNAVersionSerializerTest, ImportWhenVersionIsPresent) {
    std::stringstream sBuf;
    GNAVersionSerializer verSerializer;
    verSerializer.Export(sBuf);

    EXPECT_THAT(verSerializer.Import(sBuf), HasSubstr(ov::intel_gna::common::get_openvino_version_string()));
}

TEST(GNAVersionSerializerTest, ImportWhenVersionIsNotPresent) {
    std::stringstream sBuf;
    GNAVersionSerializer verSerializer;

    EXPECT_EQ(verSerializer.Import(sBuf).length(), 0);
}
