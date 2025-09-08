// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"

using namespace std;

int main(int argc, char** argv) {
    const string cpath_flag{"--cpath"};
    string cpath;
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++) {
        argv_vector.push_back(argv[i]);
    }
    argc = static_cast<int>(argv_vector.size());
    ::testing::InitGoogleTest(&argc, argv_vector.data());
    for (int i = 1; i < argc; i++) {
        if (cpath_flag == argv[i] && (++i) < argc) {
            cpath = argv[i];
        }
    }
    int rc = RUN_ALL_TESTS();

    return rc;
}
