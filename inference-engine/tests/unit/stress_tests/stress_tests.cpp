// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <file_utils.h>
#include <fstream>

using namespace std;

#ifdef ENABLE_STRESS_UNIT_TESTS
class StressTests : public ::testing::Test {
protected:
    const std::string DUMMY_FILE_NAME = "Dummy.txt";
    const long long BIG_FILE_SIZE = 2LL * 1024 * 1024 * 1024 + 1;

    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:

};

struct DummyFileManager {

    static void createDummyFile(const std::string &filename, const size_t size) {
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        ofs.seekp(size - 1);
        ofs.write("", 1);
    }

    static void deleteFile(const std::string &filename) {
        std::remove(filename.c_str());
    }
};

TEST_F(StressTests, checkBigFileSize) {
    DummyFileManager::createDummyFile(DUMMY_FILE_NAME, BIG_FILE_SIZE);
    long long size = FileUtils::fileSize(DUMMY_FILE_NAME);
    DummyFileManager::deleteFile(DUMMY_FILE_NAME);
    ASSERT_EQ(size, BIG_FILE_SIZE);
}
#endif //ENABLE_STRESS_UNIT_TESTS
