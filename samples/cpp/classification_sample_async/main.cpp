// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point the OpenVINO Runtime sample application
 * @file classification_sample_async/main.cpp
 * @example classification_sample_async/main.cpp
 */

#include <sys/stat.h>

#include <condition_variable>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include "classification_sample_async.h"
// clang-format on

using namespace ov::preprocess;

namespace {
bool parse_and_check_command_line(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        show_usage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        show_usage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        show_usage();
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO Runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!parse_and_check_command_line(argc, argv)) {
            return EXIT_SUCCESS;
        }

        // -------- Read input --------
        // This vector stores paths to the processed images
        std::vector<std::string> image_names;
        parseInputFilesArguments(image_names);
        if (image_names.empty())
            throw std::logic_error("No suitable images were found");

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Configure preprocessing --------
        const ov::Layout tensor_layout{"NHWC"};

        ov::preprocess::PrePostProcessor ppp(model);
        // 1) input() with no args assumes a model has a single input
        ov::preprocess::InputInfo& input_info = ppp.input();
        // 2) Set input tensor information:
        // - precision of tensor is supposed to be 'u8'
        // - layout of data is 'NHWC'
        input_info.tensor().set_element_type(ov::element::u8).set_layout(tensor_layout);
        // 3) Suppose model has 'NCHW' layout for input
        input_info.model().set_layout("NCHW");
        // 4) output() with no args assumes a model has a single result
        // - output() with no args assumes a model has a single result
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 5) Once the build() method is called, the pre(post)processing steps
        // for layout and precision conversions are inserted automatically
        model = ppp.build();

        // -------- Step 4. read input images --------
        slog::info << "Read input images" << slog::endl;

        ov::Shape input_shape = model->input().get_shape();
        const size_t width = input_shape[ov::layout::width_idx(tensor_layout)];
        const size_t height = input_shape[ov::layout::height_idx(tensor_layout)];

        std::vector<std::shared_ptr<unsigned char>> images_data;
        std::vector<std::string> valid_image_names;
        for (const auto& i : image_names) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            // Collect image data
            std::shared_ptr<unsigned char> data(reader->getData(width, height));
            if (data != nullptr) {
                images_data.push_back(data);
                valid_image_names.push_back(i);
            }
        }
        if (images_data.empty() || valid_image_names.empty())
            throw std::logic_error("Valid input images were not found!");

        // -------- Step 5. Setting batch size using image count --------
        const size_t batchSize = images_data.size();
        slog::info << "Set batch size " << std::to_string(batchSize) << slog::endl;
        ov::set_batch(model, batchSize);
        printInputAndOutputsInfo(*model);

        // -------- Step 6. Loading model to the device --------
        slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
        ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d);

        // -------- Step 7. Create infer request --------
        slog::info << "Create infer request" << slog::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Step 8. Combine multiple input images as batch --------
        ov::Tensor input_tensor = infer_request.get_input_tensor();

        for (size_t image_id = 0; image_id < images_data.size(); ++image_id) {
            const size_t image_size = shape_size(model->input().get_shape()) / batchSize;
            std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size,
                        images_data[image_id].get(),
                        image_size);
        }

        // -------- Step 9. Do asynchronous inference --------
        size_t num_iterations = 10;
        size_t cur_iteration = 0;
        std::condition_variable condVar;
        std::mutex mutex;
        std::exception_ptr exception_var;
        // -------- Step 10. Do asynchronous inference --------
        infer_request.set_callback([&](std::exception_ptr ex) {
            std::lock_guard<std::mutex> l(mutex);
            if (ex) {
                exception_var = ex;
                condVar.notify_all();
                return;
            }

            cur_iteration++;
            slog::info << "Completed " << cur_iteration << " async request execution" << slog::endl;
            if (cur_iteration < num_iterations) {
                // here a user can read output containing inference results and put new
                // input to repeat async request again
                infer_request.start_async();
            } else {
                // continue sample execution after last Asynchronous inference request
                // execution
                condVar.notify_one();
            }
        });

        // Start async request for the first time
        slog::info << "Start inference (asynchronous executions)" << slog::endl;
        infer_request.start_async();

        // Wait all iterations of the async request
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&] {
            if (exception_var) {
                std::rethrow_exception(exception_var);
            }

            return cur_iteration == num_iterations;
        });

        slog::info << "Completed async requests execution" << slog::endl;

        // -------- Step 11. Process output --------
        ov::Tensor output = infer_request.get_output_tensor();

        // Read labels from file (e.x. AlexNet.labels)
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
        constexpr size_t N_TOP_RESULTS = 10;
        ClassificationResult classificationResult(output, valid_image_names, batchSize, N_TOP_RESULTS, labels);
        classificationResult.show();
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
