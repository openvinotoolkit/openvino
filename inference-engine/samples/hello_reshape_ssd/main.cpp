// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>

#include <inference_engine.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <samples/common.hpp>
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
        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        const std::string device_name{argv[3]};
        const size_t batch_size{std::stoul(argv[4])};
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
        FormatReader::ReaderPtr reader(input_image_path.c_str());
        if (reader.get() == nullptr) {
            std::cout << "Image " + input_image_path + " cannot be read!" << std::endl;
            return 1;
        }
        size_t image_width, image_height;
        image_width = reader->width();
        image_height = reader->height();
        input_shape[0] = batch_size;
        input_shape[2] = image_height;
        input_shape[3] = image_width;
        input_shapes[input_name] = input_shape;
        std::cout << "Resizing network to the image size = [" << image_height << "x" << image_width << "] "
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
                if (op->get_type_info() == ngraph::op::DetectionOutput::get_type_info_static()) {
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
        /** Collect images data ptrs **/
        std::shared_ptr<unsigned char> image_data, original_image_data;
        /** Store image data **/
        std::shared_ptr<unsigned char> original_data(reader->getData());
        std::shared_ptr<unsigned char> data_reader(
            reader->getData(input_info->getTensorDesc().getDims()[3], input_info->getTensorDesc().getDims()[2]));
        if (data_reader.get() != nullptr) {
            original_image_data = original_data;
            image_data = data_reader;
        } else {
            throw std::logic_error("Valid input images were not found!");
        }

        /** Creating input blob **/
        Blob::Ptr image_input = infer_request.GetBlob(input_name);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        MemoryBlob::Ptr mimage = as<MemoryBlob>(image_input);
        if (!mimage) {
            std::cout << "We expect image blob to be inherited from MemoryBlob, but by fact we were not able "
                         "to cast imageInput to MemoryBlob"
                      << std::endl;
            return 1;
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto minputHolder = mimage->wmap();

        size_t num_channels = mimage->getTensorDesc().getDims()[1];
        size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

        unsigned char* data = minputHolder.as<unsigned char*>();

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < batch_size; ++image_id) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[image_id * image_size * num_channels + ch * image_size + pid] =
                        image_data.get()[pid * num_channels + ch];
                }
            }
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

        std::vector<std::vector<int>> boxes(batch_size);
        std::vector<std::vector<int>> classes(batch_size);

        /* Each detection has image_id that denotes processed image */
        for (size_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
            auto image_id = static_cast<int>(detection[cur_proposal * object_size + 0]);
            if (image_id < 0) {
                break;
            }

            float confidence = detection[cur_proposal * object_size + 2];
            auto label = static_cast<int>(detection[cur_proposal * object_size + 1]);
            auto xmin = detection[cur_proposal * object_size + 3] * image_width;
            auto ymin = detection[cur_proposal * object_size + 4] * image_height;
            auto xmax = detection[cur_proposal * object_size + 5] * image_width;
            auto ymax = detection[cur_proposal * object_size + 6] * image_height;

            if (confidence > 0.5f) {
                /** Drawing only objects with >50% probability **/
                classes[image_id].push_back(label);
                boxes[image_id].push_back(static_cast<int>(xmin));
                boxes[image_id].push_back(static_cast<int>(ymin));
                boxes[image_id].push_back(static_cast<int>(xmax - xmin));
                boxes[image_id].push_back(static_cast<int>(ymax - ymin));

                std::cout << "[" << cur_proposal << "," << label << "] element, prob = " << confidence << ", bbox = ("
                          << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                          << ", batch id = " << image_id << std::endl;
            }
        }

        for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
            addRectangles(original_image_data.get(),
                          image_height,
                          image_width,
                          boxes[batch_id],
                          classes[batch_id],
                          BBOX_THICKNESS);
            const std::string image_path = "hello_reshape_ssd_output.bmp";
            if (writeOutputBmp(image_path, original_image_data.get(), image_height, image_width)) {
                std::cout << "The resulting image was saved in the file: " + image_path << std::endl;
            } else {
                throw std::logic_error(std::string("Can't create a file: ") + image_path);
            }
        }
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
