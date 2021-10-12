// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point the Inference Engine sample application
 * @file classification_sample_async/main.cpp
 * @example classification_sample_async/main.cpp
 */

#include <format_reader_ptr.h>
#include <samples/classification_results.h>
#include <sys/stat.h>

#include <condition_variable>
#include <fstream>
#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "classification_sample_async.h"
#include "openvino/core/except.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::preprocess;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_nt <= 0) {
        throw std::logic_error("Incorrect value for nt argument. It should be greater than 0.");
    }

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        showUsage();
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get Inference Engine version
        // ------------------------------------------------------
        slog::info << "InferenceEngine: " << ov::get_openvino_version() << slog::endl;

        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // ------------------------------ Read input
        // -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty())
            throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        slog::info << "Loading OpenVINO runtime" << slog::endl;
        ov::runtime::Core ie;
        // ------------------------------ Get Available Devices
        // ------------------------------------------------------
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.get_versions(FLAGS_d) << std::endl;

        if (!FLAGS_l.empty()) {
            // Custom CPU extension is loaded as a shared library and passed as a
            // pointer to base extension
            auto extension_ptr = std::make_shared<InferenceEngine::Extension>(FLAGS_l);
            ie.add_extension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty() && (FLAGS_d == "GPU" || FLAGS_d == "MYRIAD" || FLAGS_d == "HDDL")) {
            // Config for device plugin custom extension is loaded from an .xml
            // description
            ie.set_config({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, FLAGS_d);
            slog::info << "Config for " << FLAGS_d << " device plugin custom extension loaded: " << FLAGS_c
                       << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;

        /** Read model model **/
        std::shared_ptr<ov::Function> model = ie.read_model(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------

        ov::Layout tensorLayout{"NCHW"};
        model = PrePostProcessor().
            input(InputInfo().
                tensor(InputTensorInfo().
                    set_element_type(ov::element::u8).
                    set_layout(tensorLayout)).
                network(InputNetworkInfo().
                    set_layout("NCHW"))). // model is supposed in NCHW format
            output(OutputInfo().
                tensor(OutputTensorInfo().
                    set_element_type(ov::element::f32))).
        build(model);

        // --------------------------- Prepare input tensor
        // -----------------------------------------------------
        slog::info << "Preparing input tensor" << slog::endl;

        ov::Shape input_shape = model->input().get_shape();
        const size_t width = input_shape[ov::layout::width(tensorLayout)];
        const size_t height = input_shape[ov::layout::height(tensorLayout)];

        std::vector<std::shared_ptr<unsigned char>> imagesData = {};
        std::vector<std::string> validImageNames = {};
        for (const auto& i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(reader->getData(width, height));
            if (data != nullptr) {
                imagesData.push_back(data);
                validImageNames.push_back(i);
            }
        }
        if (imagesData.empty() || validImageNames.empty())
            throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        const size_t batchSize = imagesData.size();
        input_shape[ov::layout::batch(tensorLayout)] = batchSize;
        // TODO: model->input().get_any_name()
        model->reshape({ { *model->input().get_tensor().get_names().begin(), input_shape } });
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading model to the device
        // ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ov::runtime::ExecutableNetwork executable_network = ie.compile_model(model, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create infer request
        // -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        /** Filling input tensor with images with BGR **/
        const size_t image_size = shape_size(input_shape) / batchSize;

        const std::string inputName = model->get_parameters()[0]->get_friendly_name();
        ov::runtime::Tensor input_tensor = infer_request.get_tensor(inputName);

        // /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
            std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size,
                imagesData[image_id].get(), image_size);
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // ---------------------------------------------------------
        size_t numIterations = 10;
        size_t curIteration = 0;
        std::condition_variable condVar;

        infer_request.set_callback([&] (std::exception_ptr ex) {
            if (ex)
                throw ex;

            curIteration++;
            slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
            if (curIteration < numIterations) {
                /* here a user can read output containing inference results and put new
                   input to repeat async request again */
                infer_request.start_async();
            } else {
                /* continue sample execution after last Asynchronous inference request
                 * execution */
                condVar.notify_one();
            }
        });

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        infer_request.start_async();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] {
            return curIteration == numIterations;
        });

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // TODO: get_output_tensor
        ov::runtime::Tensor output = infer_request.get_tensor(
            model->get_result()->input_value(0).get_node_shared_ptr()->get_friendly_name());

        /** Validating -nt value **/
        const size_t resultsCnt = output.get_size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this model (-nt should be less than "
                       << resultsCnt + 1 << " and more than 0)\n            Maximal value " << resultsCnt
                       << " will be used." << slog::endl;
            FLAGS_nt = resultsCnt;
        }

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;

        std::ifstream inputFile;
        inputFile.open(labelFileName, std::ios::in);
        if (inputFile.is_open()) {
            std::string strLine;
            while (std::getline(inputFile, strLine)) {
                trim(strLine);
                labels.push_back(strLine);
            }
        }
        // Prints formatted classification results
        ClassificationResult classificationResult(output, validImageNames, batchSize, FLAGS_nt, labels);
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl
               << "This sample is an API example, for any performance measurements "
                  "please use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}
