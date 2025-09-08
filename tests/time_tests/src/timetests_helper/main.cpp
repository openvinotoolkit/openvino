// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cli.h"
#include "statistics_writer.h"
#include "reshape_utils.h"
#include "timetests_helper/timer.h"

#include <iostream>


int runPipeline(const std::string &model, const std::string &device, const bool isCacheEnabled,
                const std::string &inputPrecision, const std::string &outputPrecision,
                std::map<std::string, ov::PartialShape> reshapeShapes,
                std::map<std::string, std::vector<size_t>> dataShapes);

/**
 * @brief Parses command line and check required arguments
 */
bool parseAndCheckCommandLine(int argc, char **argv) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_m.empty())
        throw std::logic_error(
                "Model is required but not set. Please set -m option.");

    if (FLAGS_d.empty())
        throw std::logic_error(
                "Device is required but not set. Please set -d option.");

    if (FLAGS_s.empty())
        throw std::logic_error(
                "Statistics file path is required but not set. Please set -s option.");

    return true;
}

/**
 * @brief Function calls `runPipeline` with mandatory time tracking of full run
 */
int _runPipeline(std::map<std::string, ov::PartialShape> dynamicShapes,
                 std::map<std::string, std::vector<size_t>> staticShapes) {
    SCOPED_TIMER(full_run);
    return runPipeline(FLAGS_m, FLAGS_d, FLAGS_c, FLAGS_ip, FLAGS_op, dynamicShapes, staticShapes);
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv) {
    if (!parseAndCheckCommandLine(argc, argv))
        return -1;

    auto dynamicShapes = parseReshapeShapes(FLAGS_reshape_shapes);
    auto staticShapes = parseDataShapes(FLAGS_data_shapes);

    auto status =  _runPipeline(dynamicShapes, staticShapes);
    StatisticsWriter::Instance().setFile(FLAGS_s);
    StatisticsWriter::Instance().write();
    return status;
}
