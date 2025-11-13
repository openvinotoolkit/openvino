// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"

#include "model_creation_sample.hpp"
// clang-format on

constexpr auto N_TOP_RESULTS = 1;
constexpr auto LENET_WEIGHTS_SIZE = 1724336;
constexpr auto LENET_NUM_CLASSES = 10;

using namespace ov;
using namespace ov::preprocess;

/**
 * @brief Read file to the buffer
 * @param file_name string
 * @param buffer to store file content
 * @param maxSize length of file
 * @return none
 */
void read_file(const std::string& file_name, void* buffer, size_t maxSize) {
    std::ifstream input_file;

    input_file.open(file_name, std::ios::binary | std::ios::in);
    if (!input_file.is_open()) {
        throw std::logic_error("Cannot open weights file");
    }

    if (!input_file.read(reinterpret_cast<char*>(buffer), maxSize)) {
        input_file.close();
        throw std::logic_error("Cannot read bytes from weights file");
    }

    input_file.close();
}

/**
 * @brief Read .bin file with weights for the trained model
 * @param filepath string
 * @return weightsPtr tensor blob
 */
ov::Tensor read_weights(const std::string& filepath) {
    std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);

    int64_t fileSize = weightFile.tellg();
    OPENVINO_ASSERT(fileSize == LENET_WEIGHTS_SIZE,
                    "Incorrect weights file. This sample works only with LeNet "
                    "classification model.");

    ov::Tensor weights(ov::element::u8, {static_cast<size_t>(fileSize)});
    read_file(filepath, weights.data(), weights.get_byte_size());

    return weights;
}

/**
 * @brief Create ov::Model
 * @return Ptr to ov::Model
 */
std::shared_ptr<ov::Model> create_model(const std::string& path_to_weights) {
    const ov::Tensor weights = read_weights(path_to_weights);
    const std::uint8_t* data = weights.data<std::uint8_t>();

    // -------input------
    std::vector<ptrdiff_t> padBegin{0, 0};
    std::vector<ptrdiff_t> padEnd{0, 0};

    auto paramNode = std::make_shared<ov::opset13::Parameter>(ov::element::Type_t::f32, ov::Shape({64, 1, 28, 28}));

    // -------convolution 1----
    auto convFirstShape = Shape{20, 1, 5, 5};
    auto convolutionFirstConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, convFirstShape, data);

    auto convolutionNodeFirst = std::make_shared<opset13::Convolution>(paramNode->output(0),
                                                                       convolutionFirstConstantNode->output(0),
                                                                       Strides({1, 1}),
                                                                       CoordinateDiff(padBegin),
                                                                       CoordinateDiff(padEnd),
                                                                       Strides({1, 1}));

    // -------Add--------------
    auto addFirstShape = Shape{1, 20, 1, 1};
    auto offset = shape_size(convFirstShape) * sizeof(float);
    auto addFirstConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, addFirstShape, data + offset);

    auto addNodeFirst =
        std::make_shared<opset13::Add>(convolutionNodeFirst->output(0), addFirstConstantNode->output(0));

    // -------MAXPOOL----------
    Shape padBeginShape{0, 0};
    Shape padEndShape{0, 0};

    auto maxPoolingNodeFirst = std::make_shared<opset13::MaxPool>(addNodeFirst->output(0),
                                                                  Strides{2, 2},
                                                                  Strides{1, 1},
                                                                  padBeginShape,
                                                                  padEndShape,
                                                                  Shape{2, 2},
                                                                  op::RoundingType::CEIL);

    // -------convolution 2----
    auto convSecondShape = Shape{50, 20, 5, 5};
    offset += shape_size(addFirstShape) * sizeof(float);
    auto convolutionSecondConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::f32, convSecondShape, data + offset);

    auto convolutionNodeSecond = std::make_shared<opset13::Convolution>(maxPoolingNodeFirst->output(0),
                                                                        convolutionSecondConstantNode->output(0),
                                                                        Strides({1, 1}),
                                                                        CoordinateDiff(padBegin),
                                                                        CoordinateDiff(padEnd),
                                                                        Strides({1, 1}));

    // -------Add 2------------
    auto addSecondShape = Shape{1, 50, 1, 1};
    offset += shape_size(convSecondShape) * sizeof(float);
    auto addSecondConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::f32, addSecondShape, data + offset);

    auto addNodeSecond =
        std::make_shared<opset13::Add>(convolutionNodeSecond->output(0), addSecondConstantNode->output(0));

    // -------MAXPOOL 2--------
    auto maxPoolingNodeSecond = std::make_shared<opset13::MaxPool>(addNodeSecond->output(0),
                                                                   Strides{2, 2},
                                                                   Strides{1, 1},
                                                                   padBeginShape,
                                                                   padEndShape,
                                                                   Shape{2, 2},
                                                                   op::RoundingType::CEIL);

    // -------Reshape----------
    auto reshapeFirstShape = Shape{2};
    auto reshapeOffset = shape_size(addSecondShape) * sizeof(float) + offset;
    auto reshapeFirstConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::i64, reshapeFirstShape, data + reshapeOffset);

    auto reshapeFirstNode =
        std::make_shared<opset13::Reshape>(maxPoolingNodeSecond->output(0), reshapeFirstConstantNode->output(0), true);

    // -------MatMul 1---------
    auto matMulFirstShape = Shape{500, 800};
    offset = shape_size(reshapeFirstShape) * sizeof(int64_t) + reshapeOffset;
    auto matMulFirstConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::f32, matMulFirstShape, data + offset);

    auto matMulFirstNode =
        std::make_shared<opset13::MatMul>(reshapeFirstNode->output(0), matMulFirstConstantNode->output(0), false, true);

    // -------Add 3------------
    auto addThirdShape = Shape{1, 500};
    offset += shape_size(matMulFirstShape) * sizeof(float);
    auto addThirdConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, addThirdShape, data + offset);

    auto addThirdNode = std::make_shared<opset13::Add>(matMulFirstNode->output(0), addThirdConstantNode->output(0));

    // -------Relu-------------
    auto reluNode = std::make_shared<opset13::Relu>(addThirdNode->output(0));

    // -------Reshape 2--------
    auto reshapeSecondShape = Shape{2};
    auto reshapeSecondConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::i64, reshapeSecondShape, data + reshapeOffset);

    auto reshapeSecondNode =
        std::make_shared<opset13::Reshape>(reluNode->output(0), reshapeSecondConstantNode->output(0), true);

    // -------MatMul 2---------
    auto matMulSecondShape = Shape{10, 500};
    offset += shape_size(addThirdShape) * sizeof(float);
    auto matMulSecondConstantNode =
        std::make_shared<opset13::Constant>(element::Type_t::f32, matMulSecondShape, data + offset);

    auto matMulSecondNode = std::make_shared<opset13::MatMul>(reshapeSecondNode->output(0),
                                                              matMulSecondConstantNode->output(0),
                                                              false,
                                                              true);

    // -------Add 4------------
    auto add4Shape = Shape{1, 10};
    offset += shape_size(matMulSecondShape) * sizeof(float);
    auto add4ConstantNode = std::make_shared<opset13::Constant>(element::Type_t::f32, add4Shape, data + offset);

    auto add4Node = std::make_shared<opset13::Add>(matMulSecondNode->output(0), add4ConstantNode->output(0));

    // -------softMax----------
    auto softMaxNode = std::make_shared<opset13::Softmax>(add4Node->output(0), 1);
    softMaxNode->get_output_tensor(0).set_names({"output_tensor"});

    // ------- OpenVINO function--
    auto result_full = std::make_shared<opset13::Result>(softMaxNode->output(0));

    std::shared_ptr<ov::Model> fnPtr =
        std::make_shared<ov::Model>(result_full, ov::ParameterVector{paramNode}, "lenet");

    return fnPtr;
}

/**
 * @brief The entry point for OpenVINO ov::Model creation sample
 */
int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 3) {
            std::cout << "Usage : " << argv[0] << " <path_to_lenet_weights> <device>" << std::endl;
            return EXIT_FAILURE;
        }
        const std::string weights_path{argv[1]};
        const std::string device_name{argv[2]};

        // -------- Step 1. Initialize OpenVINO Runtime Core object --------
        ov::Core core;

        slog::info << "Device info: " << slog::endl;
        slog::info << core.get_versions(device_name) << slog::endl;

        // -------- Step 2. Create network using ov::Function --------
        slog::info << "Create model from weights: " << weights_path << slog::endl;
        std::shared_ptr<ov::Model> model = create_model(weights_path);
        printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Incorrect number of inputs for LeNet");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Incorrect number of outputs for LeNet");

        ov::Shape input_shape = model->input().get_shape();
        OPENVINO_ASSERT(input_shape.size() == 4, "Incorrect input dimensions for LeNet");

        const ov::Shape output_shape = model->output().get_shape();
        OPENVINO_ASSERT(output_shape.size() == 2, "Incorrect output dimensions for LeNet");

        const auto classCount = output_shape[1];
        OPENVINO_ASSERT(classCount <= LENET_NUM_CLASSES, "Incorrect number of output classes for LeNet model");

        // -------- Step 3. Apply preprocessing --------
        const Layout tensor_layout{"NHWC"};

        // apply preprocessing
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        // 1) InputInfo() with no args assumes a model has a single input
        ov::preprocess::InputInfo& input_info = ppp.input();
        // 2) Set input tensor information:
        // - layout of data is 'NHWC'
        // - precision of tensor is supposed to be 'u8'
        input_info.tensor().set_layout(tensor_layout).set_element_type(element::u8);
        // 3) Here we suppose model has 'NCHW' layout for input
        input_info.model().set_layout("NCHW");

        // 4) Once the build() method is called, the preprocessing steps
        // for layout and precision conversions are inserted automatically
        model = ppp.build();

        // Set batch size using images count
        const size_t batch_size = digits.size();

        // -------- Step 4. Reshape a model to new batch size --------
        // Setting batch size using image count
        ov::set_batch(model, batch_size);
        slog::info << "Batch size is " << std::to_string(batch_size) << slog::endl;
        printInputAndOutputsInfo(*model);

        // -------- Step 5. Compiling model for the device --------
        slog::info << "Compiling a model for the " << device_name << " device" << slog::endl;
        ov::CompiledModel compiled_model = core.compile_model(model, device_name);

        // -------- Step 6. Create infer request --------
        slog::info << "Create infer request" << slog::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Step 7. Combine multiple input images as batch --------
        slog::info << "Combine images in batch and set to input tensor" << slog::endl;
        ov::Tensor input_tensor = infer_request.get_input_tensor();

        // Iterate over all input images and copy data to input tensor
        for (size_t image_id = 0; image_id < digits.size(); ++image_id) {
            const size_t image_size = shape_size(model->input().get_shape()) / batch_size;
            std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size, digits[image_id], image_size);
        }

        // -------- Step 8. Do sync inference --------
        slog::info << "Start sync inference" << slog::endl;
        infer_request.infer();

        // -------- Step 9. Process output --------
        slog::info << "Processing output tensor" << slog::endl;
        const ov::Tensor output_tensor = infer_request.get_output_tensor();

        const std::vector<std::string> lenet_labels{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

        // Prints formatted classification results
        ClassificationResult classification_result(output_tensor,
                                                   lenet_labels,  // in this sample images have the same names as labels
                                                   batch_size,
                                                   N_TOP_RESULTS,
                                                   lenet_labels);
        classification_result.show();
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
