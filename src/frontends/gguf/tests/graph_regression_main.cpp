// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <string>

#include "graph_regression_utils.hpp"
#include "gtest/gtest.h"

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--gguf-snapshot") {
        try {
            return ov_gguf_test::graph_snapshot_main(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "GGUF graph snapshot failed: " << error.what() << '\n';
            return 1;
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
