// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <iostream>

#include "timetests_helper/timer.h"
using namespace InferenceEngine;

/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device) {
  auto pipeline = [](const std::string &model, const std::string &device) {
    SCOPED_TIMER(first_time_to_inference);

    Core ie;
    CNNNetwork cnnNetwork;
    ExecutableNetwork exeNetwork;

    {
      SCOPED_TIMER(read_network);
      cnnNetwork = ie.ReadNetwork(model);
    }

    {
      SCOPED_TIMER(load_network);
      ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
    }
  };

  try {
    pipeline(model, device);
  } catch (const InferenceEngine::details::InferenceEngineException &iex) {
    std::cerr
        << "Inference Engine pipeline failed with Inference Engine exception:\n"
        << iex.what();
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "Inference Engine pipeline failed with exception:\n"
              << ex.what();
    return 2;
  } catch (...) {
    std::cerr << "Inference Engine pipeline failed\n";
    return 3;
  }
  return 0;
}