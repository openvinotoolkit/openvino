// Copyright (C) 2018-2022 Intel Corporation
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

constexpr auto N_TOP_RESULTS = 10;

using namespace ov::preprocess;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
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

static inline ov::Layout getLayoutFromShape(const ov::Shape& shape) {
    if (shape.size() == 2) {
        return "NC";
    } else if (shape.size() == 3) {
        return (shape[0] >= 1 && shape[0] <= 4) ? "CHW" : "HWC";
    } else if (shape.size() == 4) {
        return (shape[1] >= 1 && shape[1] <= 4) ? "NCHW" : "NHWC";
    } else {
        throw std::runtime_error("Usupported " + std::to_string(shape.size()) + "D shape");
    }
}

template <class T>
static inline void CopyToTensor(const std::uint8_t* pSourceImg,
                                T* pTensor,
                                const size_t width,
                                const size_t height,
                                const size_t channels) {
    for (int p = 0; p < width * height * channels; p++) {
        pTensor[p] = (T)pSourceImg[p];
    }
}

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

        // -------- Step 2. Produce a compiled model --------
        ov::CompiledModel compiled_model;
        bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";

        // Set batch size to the the number of images passed with -i
        const size_t batchSize = image_names.size();

        if (!isNetworkCompiled) {
            slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
            std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
            printInputAndOutputsInfo(*model);

            OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
            OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

            // -------- Step 2.a Configure preprocessing --------
            ov::preprocess::PrePostProcessor ppp(model);
            // 1) input() with no args assumes a model has a single input
            ov::preprocess::InputInfo& input_info = ppp.input();
            // 2) Set input tensor information:
            // - precision of tensor is supposed to be 'u8'
            // - layout of data is 'NHWC'
            input_info.tensor().set_element_type(ov::element::u8).set_layout("NHWC");
            // 3) Here we suppose model has 'NCHW' layout for input
            input_info.model().set_layout("NCHW");
            // 4) output() with no args assumes a model has a single result
            // - output() with no args assumes a model has a single result
            // - precision of tensor is supposed to be 'f32'
            ppp.output().tensor().set_element_type(ov::element::f32);

            // 5) Once the build() method is called, the pre(post)processing steps
            // for layout and precision conversions are inserted automatically
            model = ppp.build();

            slog::info << "Set model batch size " << std::to_string(batchSize) << slog::endl;
            ov::set_batch(model, batchSize);
            printInputAndOutputsInfo(*model);

            // -------- Step 2.b Produce ov::CompiledModel from ov::Model --------
            slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
            
            compiled_model = core.compile_model(model, FLAGS_d);
        }
        else {
            slog::info << "Loading pre-compiled blob:" << slog::endl << FLAGS_m << slog::endl;

            // --------  Produce ov::CompiledModel by importing pre-compiled blob from disk --------
            std::ifstream modelStream(FLAGS_m, std::ios_base::binary | std::ios_base::in);
            if (!modelStream.is_open()) {
                throw std::runtime_error("Cannot open model file " + FLAGS_m);
            }
            compiled_model = core.import_model(modelStream, FLAGS_d, {});
            modelStream.close();
        }
        
        // -------- Step 3. read input images --------
        slog::info << "Read input images" << slog::endl;

        ov::Shape input_shape = compiled_model.input().get_shape();
        
        // ------- Double check that the batch of the compiled model matches 'batchSize' ------
        // Note that in case of using a pre-compiled blob, the batch size was set at compile-time.
        const size_t input_batch_size = input_shape[ov::layout::batch_idx(getLayoutFromShape(input_shape))];
        if (input_batch_size != batchSize) {
            throw std::runtime_error("Batch size of compiled model input (" + std::to_string(input_batch_size) + ")" +
                                     " doesn't number of images passed with -i (" + std::to_string(batchSize) + ")");
        }

        const size_t width = input_shape[ov::layout::width_idx(getLayoutFromShape(input_shape))];
        const size_t height = input_shape[ov::layout::height_idx(getLayoutFromShape(input_shape))];
        const size_t channels = input_shape[ov::layout::channels_idx(getLayoutFromShape(input_shape))];

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

        // -------- Step 4. Create infer request --------
        slog::info << "Create infer request" << slog::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Step 5. Combine multiple input images as batch --------
        ov::Tensor input_tensor = infer_request.get_input_tensor();
        ov::element::Type input_type = input_tensor.get_element_type();

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        ov::element::Type output_type = output_tensor.get_element_type();

        slog::info << "\tinputs " << slog::endl;
        slog::info << "\t\tinput type: " << input_type << slog::endl;
        slog::info << "\t\tinput shape: " << input_tensor.get_shape() << slog::endl;
        slog::info << "\toutputs " << slog::endl;
        slog::info << "\t\toutput type: " << output_type << slog::endl;
        slog::info << "\t\toutput shape: " << output_tensor.get_shape() << slog::endl;

        for (size_t image_id = 0; image_id < images_data.size(); ++image_id) {
            const size_t image_size = shape_size(compiled_model.input().get_shape()) / batchSize;
            switch (input_type) {
#define TENSOR_COPY_CASE(elem_type)                                                                          \
                case ov::element::Type_t::elem_type: {                                                       \
                    using tensor_type = ov::fundamental_type_for<ov::element::Type_t::elem_type>;            \
                    tensor_type* pTensor = input_tensor.data<tensor_type>() + image_id * image_size;         \
                    CopyToTensor<tensor_type>(images_data[image_id].get(), pTensor, width, height, channels);\
                    break;                                                                                   \
                }
                TENSOR_COPY_CASE(f32);
                TENSOR_COPY_CASE(f64);
                TENSOR_COPY_CASE(f16);
                TENSOR_COPY_CASE(i16);
                TENSOR_COPY_CASE(u8);
                TENSOR_COPY_CASE(i8);
                TENSOR_COPY_CASE(u16);
                TENSOR_COPY_CASE(i32);
                TENSOR_COPY_CASE(u32);
                TENSOR_COPY_CASE(i64);
                TENSOR_COPY_CASE(u64);
                default:
                    throw std::runtime_error("Unsupported input tensor type for image copy");
            } 
        }

        // -------- Step 6. Do asynchronous inference --------
        size_t num_iterations = 10;
        size_t cur_iteration = 0;
        std::condition_variable condVar;
        std::mutex mutex;
        std::exception_ptr exception_var;
        // -------- Step 10. Do asynchronous inference --------
        infer_request.set_callback([&](std::exception_ptr ex) {
            if (ex) {
                exception_var = ex;
                condVar.notify_all();
                return;
            }

            std::lock_guard<std::mutex> l(mutex);
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

        // -------- Step 7. Process output --------
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
