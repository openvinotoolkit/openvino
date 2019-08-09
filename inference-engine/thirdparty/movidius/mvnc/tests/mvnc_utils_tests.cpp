// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ncCommPrivate.h"
#include "mvnc_tests_common.hpp"
#include <fstream>

extern "C" {
#include "mvStringUtils.h"
}

class MvncUtilsTest : public ::testing::Test {
public:
    void TearDown() override {
        std::remove(mvcmdExpectedPath.c_str());
    }

protected:
    std::string mvcmdExpectedPath = "";
    // FIXME: seems it is not going to work on Windows
    const std::string tmpDir = "/tmp";
};

TEST_F(MvncUtilsTest, CanGetSpecialFWIfUniversalIsNotPresent) {
    mvcmdExpectedPath = tmpDir + "/MvNCAPI-ma2480.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    mv_strcpy(mvcmdFilePath, MAX_PATH, tmpDir.c_str());

    const char *dummyDevAddr2480 = "0-ma2480";

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, dummyDevAddr2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}

TEST_F(MvncUtilsTest, CanGetUniversalFWIfItExists) {
    mvcmdExpectedPath = tmpDir + "/MvNCAPI-ma2x8x.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    mv_strcpy(mvcmdFilePath, MAX_PATH, tmpDir.c_str());

    const char *dummyDevAddr2480 = "0-ma2480";

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, dummyDevAddr2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}
