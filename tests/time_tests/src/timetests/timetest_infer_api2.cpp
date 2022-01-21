// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>

#include <fstream>

#include "common_utils.h"
#include "reshape_utils.h"
#include "timetests_helper/timer.h"
#include "timetests_helper/utils.h"


/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device, const bool isCacheEnabled,
                std::map<std::string, ov::PartialShape> reshapeShapes,
                std::map<std::string, std::vector<size_t>> dataShapes) {
  auto pipeline = [](const std::string &model, const std::string &device, const bool isCacheEnabled,
                     std::map<std::string, ov::PartialShape> reshapeShapes,
                     std::map<std::string, std::vector<size_t>> dataShapes) {
    ov::runtime::Core ie;
    std::shared_ptr<ov::Model> cnnNetwork;
    ov::runtime::CompiledModel exeNetwork;
    ov::runtime::InferRequest inferRequest;

    bool reshape = false;
    if (!reshapeShapes.empty()) {
      reshape = true;
    }

    // first_inference_latency = time_to_inference + first_inference
    {
      SCOPED_TIMER(time_to_inference);
      {
        SCOPED_TIMER(load_plugin);
        TimeTest::setPerformanceConfigAPI2(ie, device);
        ie.get_versions(device);

        if (isCacheEnabled)
          ie.set_config({{CONFIG_KEY(CACHE_DIR), "models_cache"}});
      }
      {
        SCOPED_TIMER(create_exenetwork);
        if (!isCacheEnabled) {
          if (TimeTest::fileExt(model) == "blob") {
            SCOPED_TIMER(import_network);
            std::ifstream streamModel{model};
            exeNetwork = ie.import_model(streamModel, device);
          }
          else {
            {
              SCOPED_TIMER(read_network);
              cnnNetwork = ie.read_model(model);
            }
            if (reshape) {
              {
                SCOPED_TIMER(reshape);
                // dynamic input 1..2 3 600 600 not reshapable
                cnnNetwork->reshape(reshapeShapes);
              }
            }
            {
              SCOPED_TIMER(load_network);
              exeNetwork = ie.compile_model(cnnNetwork, device);
            }
          }
        }
        else {
          SCOPED_TIMER(load_network_cache);
          exeNetwork = ie.compile_model(model, device);
        }
      }
      inferRequest = exeNetwork.create_infer_request();
    }
    {
      SCOPED_TIMER(first_inference);
      {
        SCOPED_TIMER(fill_inputs);
        std::vector<ov::Output<const ov::Node>> inputs = exeNetwork.inputs();

        // data input should be NHWC if model NCHW
        for (size_t i = 0; i < inputs.size(); i++) {
          std::string name;
          try {
            name = inputs[i].get_any_name();
            std::cerr << "name: " << name << "\n";
          } catch (const ov::Exception &iex) {
            // Attempt to get a name for a Tensor without names
          }

          if (!inputs[i].get_partial_shape().is_dynamic()) {
            std::cerr << "name: " << name << " not dynamic: " << inputs[i].get_partial_shape() << "\n";
          }
          else {
            std::cerr << "name: " << name << " is dynamic: " << inputs[i].get_partial_shape() << "\n";
          }
        }

        if (reshape) {
          fillTensorsDynamic(inferRequest, inputs, dataShapes);
        } else {
          fillTensors(inferRequest, inputs);
        }
      }
      inferRequest.infer();
    }
  };

  try {
    pipeline(model, device, isCacheEnabled, reshapeShapes, dataShapes);
  } catch (const ov::Exception &iex) {
    std::cerr
        << "OpenVINO runtime pipeline failed with OpenVINO runtime exception:\n"
        << iex.what();
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "OpenVINO runtime pipeline failed with exception:\n"
              << ex.what();
    return 2;
  } catch (...) {
    std::cerr << "Inference Engine pipeline failed\n";
    return 3;
  }
  return 0;
}
