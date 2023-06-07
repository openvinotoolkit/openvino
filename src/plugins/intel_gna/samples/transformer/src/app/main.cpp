// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logger/logger.hpp"
#include "menu.hpp"

using namespace transformation_sample;

int main(int argc, char* argv[]) {
    try {
        log() << "Starting transforming application" << std::endl;
        Menu menu(argc, argv);
        menu.execute_action();
    } catch (const std::exception& exc) {
        log_error() << "Issue when running application: " << exc.what() << std::endl;
        return -1;
    }

    return 0;
}