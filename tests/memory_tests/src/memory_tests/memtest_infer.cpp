// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <iostream>

#include "common.h"
#include "memory_tests_helper/memory_counter.h"
#include "memory_tests_helper/utils.h"
using namespace InferenceEngine;


/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device) {
  auto pipeline = [](const std::string &model, const std::string &device) {
    Core ie;
    CNNNetwork cnnNetwork;
    ExecutableNetwork exeNetwork;
    InferRequest inferRequest;
    size_t batchSize = 0;

    {
      SCOPED_MEM_COUNTER(full_run);
      {
        SCOPED_MEM_COUNTER(first_inference_latency);
        {
          SCOPED_MEM_COUNTER(load_plugin);
          ie.GetVersions(device);
        }
        {
          SCOPED_MEM_COUNTER(create_exenetwork);
          if (MemoryTest::fileExt(model) == "blob") {
            SCOPED_MEM_COUNTER(import_network);
            exeNetwork = ie.ImportNetwork(model, device);
          }
          else {
            {
              SCOPED_MEM_COUNTER(read_network);
              cnnNetwork = ie.ReadNetwork(model);
              batchSize = cnnNetwork.getBatchSize();
            }

            {
              SCOPED_MEM_COUNTER(load_network);
              exeNetwork = ie.LoadNetwork(cnnNetwork, device);
            }
          }
        }
      }

      {
        SCOPED_MEM_COUNTER(first_inference);
        inferRequest = exeNetwork.CreateInferRequest();

        {
          SCOPED_MEM_COUNTER(fill_inputs)
          batchSize = batchSize != 0 ? batchSize : 1;
          const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
          fillBlobs(inferRequest, inputsInfo, batchSize);
        }
        inferRequest.Infer();
      }
    }
  };

  try {
    pipeline(model, device);
  } catch (const InferenceEngine::Exception &iex) {
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
