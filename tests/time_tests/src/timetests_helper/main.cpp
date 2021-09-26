// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cli.h"
#include "statistics_writer.h"
#include "timetests_helper/timer.h"

#include <iostream>

int runPipeline(const std::string &model, const std::string &device);

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
int _runPipeline() {
  SCOPED_TIMER(full_run);
  return runPipeline(FLAGS_m, FLAGS_d);
}

/**
 * @brief Main entry point
 */
int main(int argc, char **argv) {
  if (!parseAndCheckCommandLine(argc, argv))
    return -1;

  auto status =  _runPipeline();
  StatisticsWriter::Instance().setFile(FLAGS_s);
  StatisticsWriter::Instance().write();
  return status;
}