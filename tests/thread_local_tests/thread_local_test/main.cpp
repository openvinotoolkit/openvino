// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <thread>
#include <future>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
