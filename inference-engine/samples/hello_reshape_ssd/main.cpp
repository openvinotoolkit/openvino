// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <vector>

#include "reshape_ssd_extension.hpp"

using namespace InferenceEngine;

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (argc != 5) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device> <batch>" << std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_model {argv[1]};
        const std::string input_image_path {argv[2]};
        const std::string device_name {argv[3]};
        const size_t batch_size {std::stoul(argv[4])};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        Core ie;

        IExtensionPtr inPlaceExtension;
        if (device_name.find("CPU") != std::string::npos) {
            inPlaceExtension = std::make_shared<InPlaceExtension>();
            // register sample's custom kernel (CustomReLU)
            ie.AddExtension(inPlaceExtension);
        }
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);

        OutputsDataMap outputs_info(network.getOutputsInfo());
        InputsDataMap inputs_info(network.getInputsInfo());
        if (inputs_info.size() != 1 || outputs_info.size() != 1)
            throw std::logic_error("Sample supports clean SSD network with one input and one output");

        // --------------------------- Resize network to match image sizes and given
        // batch----------------------
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        cv::Mat image = cv::imread(input_image_path);
        input_shape[0] = batch_size;
        input_shape[2] = static_cast<size_t>(image.rows);
        input_shape[3] = static_cast<size_t>(image.cols);
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" << image.rows << "x" << image.cols << "] "
                  << "with batch = " << batch_size << std::endl;
        network.reshape(input_shapes);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        InputInfo::Ptr input_info;
        std::tie(input_name, input_info) = *inputs_info.begin();
        // Set input layout and precision
        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);
        // --------------------------- Prepare output blobs
        // ----------------------------------------------------
        DataPtr output_info;
        std::string output_name;
        std::tie(output_name, output_info) = *outputs_info.begin();
        // SSD has an additional post-processing DetectionOutput layer
        // that simplifies output filtering, try to find it.
        if (auto ngraphFunction = network.getFunction()) {
            for (const auto& op : ngraphFunction->get_ops()) {
                if (op->get_type_info() == ngraph::op::DetectionOutput::type_info) {
                    if (output_info->getName() != op->get_friendly_name()) {
                        throw std::logic_error("Detection output op does not produce a network output");
                    }
                    break;
                }
            }
        }

        const SizeVector output_shape = output_info->getTensorDesc().getDims();
        const size_t max_proposal_count = output_shape[2];
        const size_t object_size = output_shape[3];
        if (object_size != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }
        if (output_shape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }
        if (output_info == nullptr) {
            IE_THROW() << "[SAMPLES] internal error - output information is empty";
        }

        output_info->setPrecision(Precision::FP32);

        auto dumpVec = [](const SizeVector& vec) -> std::string {
            if (vec.empty())
                return "[]";
            std::stringstream oss;
            oss << "[" << vec[0];
            for (size_t i = 1; i < vec.size(); i++)
                oss << "," << vec[i];
            oss << "]";
            return oss.str();
        };
        std::cout << "Resulting input shape = " << dumpVec(input_shape) << std::endl;
        std::cout << "Resulting output shape = " << dumpVec(output_shape) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading a model to the device
        // ------------------------------------------
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create an infer request
        // -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        Blob::Ptr input = infer_request.GetBlob(input_name);
        for (size_t b = 0; b < batch_size; b++) {
            matU8ToBlob<uint8_t>(image, input, b);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // --------------------------------------------------------
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto moutputHolder = moutput->rmap();
        const float* detection = moutputHolder.as<const float*>();

        /* Each detection has image_id that denotes processed image */
        for (size_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
            float image_id = detection[cur_proposal * object_size + 0];
            float label = detection[cur_proposal * object_size + 1];
            float confidence = detection[cur_proposal * object_size + 2];
            /* CPU and GPU devices have difference in DetectionOutput layer, so we
             * need both checks */
            if (image_id < 0 || confidence == 0.0f) {
                continue;
            }

            float xmin = detection[cur_proposal * object_size + 3] * image.cols;
            float ymin = detection[cur_proposal * object_size + 4] * image.rows;
            float xmax = detection[cur_proposal * object_size + 5] * image.cols;
            float ymax = detection[cur_proposal * object_size + 6] * image.rows;

            if (confidence > 0.5f) {
                /** Drawing only objects with >50% probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << confidence;
                cv::rectangle(image, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 0, 255));
                std::cout << "[" << cur_proposal << "," << label << "] element, prob = " << confidence << ", bbox = (" << xmin << "," << ymin << ")-(" << xmax
                          << "," << ymax << ")"
                          << ", batch id = " << image_id << std::endl;
            }
        }

        cv::imwrite("hello_reshape_ssd_output.jpg", image);
        std::cout << "The resulting image was saved in the file: "
                     "hello_reshape_ssd_output.jpg"
                  << std::endl;
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << std::endl
              << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
