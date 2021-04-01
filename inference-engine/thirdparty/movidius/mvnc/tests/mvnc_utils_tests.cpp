// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ncCommPrivate.h"
#include "mvnc_test_helper.h"

#include <gtest/gtest.h>
#include <fstream>

extern "C" {
#include "XLinkStringUtils.h"
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
    mvcmdExpectedPath = tmpDir + "/usb-ma248x.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    mv_strcpy(mvcmdFilePath, MAX_PATH, tmpDir.c_str());

    deviceDesc_t dummyDevDesc2480;
    strcpy(dummyDevDesc2480.name, "0-ma2480");
    dummyDevDesc2480.protocol = X_LINK_USB_VSC;
    dummyDevDesc2480.platform = X_LINK_MYRIAD_X;

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, MAX_PATH, dummyDevDesc2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}

TEST_F(MvncUtilsTest, CanGetUniversalFWIfItExists) {
    mvcmdExpectedPath = tmpDir + "/usb-ma2x8x.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    mv_strcpy(mvcmdFilePath, MAX_PATH, tmpDir.c_str());

    deviceDesc_t dummyDevDesc2480;
    strcpy(dummyDevDesc2480.name, "0-ma2480");
    dummyDevDesc2480.protocol = X_LINK_USB_VSC;
    dummyDevDesc2480.platform = X_LINK_MYRIAD_X;

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, MAX_PATH, dummyDevDesc2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}
