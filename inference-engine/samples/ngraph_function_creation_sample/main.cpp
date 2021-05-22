// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>
#include <gflags/gflags.h>
#include <samples/classification_results.h>

#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "ngraph_function_creation_sample.hpp"

using namespace InferenceEngine;
using namespace ngraph;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_nt <= 0 || FLAGS_nt > 10) {
        throw std::logic_error("Incorrect value for nt argument. It should be "
                               "greater than 0 and less than 10.");
    }

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Path to a .bin file with weights for the trained model is required "
                               "but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        showUsage();
        throw std::logic_error("Path to an image is required but not set. Please set -i option.");
    }

    return true;
}

/**
 * @brief Read file to the buffer
 * @param file_name string
 * @param buffer to store file content
 * @param maxSize length of file
 * @return none
 */
void readFile(const std::string& file_name, void* buffer, size_t maxSize) {
    std::ifstream inputFile;

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) {
        throw std::logic_error("Cannot open weights file");
    }

    if (!inputFile.read(reinterpret_cast<char*>(buffer), maxSize)) {
        inputFile.close();
        throw std::logic_error("Cannot read bytes from weights file");
    }

    inputFile.close();
}

/**
 * @brief Read .bin file with weights for the trained model
 * @param filepath string
 * @return weightsPtr tensor blob
 */
TBlob<uint8_t>::CPtr ReadWeights(std::string filepath) {
    std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);
    int64_t fileSize = weightFile.tellg();

    if (fileSize < 0) {
        throw std::logic_error("Incorrect weights file");
    }

    size_t ulFileSize = static_cast<size_t>(fileSize);

    TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>({Precision::FP32, {ulFileSize}, Layout::C}));
    weightsPtr->allocate();
    readFile(filepath, weightsPtr->buffer(), ulFileSize);

    return weightsPtr;
}

/**
 * @brief Create ngraph function
 * @return Ptr to ngraph function
 */
std::shared_ptr<Function> createNgraphFunction() {
    TBlob<uint8_t>::CPtr weightsPtr = ReadWeights(FLAGS_m);

    if (weightsPtr->byteSize() != 1724336)
        IE_THROW() << "Incorrect weights file. This sample works only with LeNet "
                      "classification network.";

    // -------input------
    std::vector<ptrdiff_t> padBegin {0, 0};
    std::vector<ptrdiff_t> padEnd {0, 0};

    auto paramNode = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t> {{64, 1, 28, 28}}));
    paramNode->set_friendly_name("Parameter");

    // -------convolution 1----
    auto convFirstShape = Shape {20, 1, 5, 5};
    std::shared_ptr<Node> convolutionFirstConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, convFirstShape, weightsPtr->cbuffer().as<uint8_t*>());

    std::shared_ptr<Node> convolutionNodeFirst =
        std::make_shared<op::v1::Convolution>(paramNode->output(0), convolutionFirstConstantNode->output(0), Strides(SizeVector {1, 1}),
                                              CoordinateDiff(padBegin), CoordinateDiff(padEnd), Strides(SizeVector {1, 1}));

    // -------Add--------------
    auto addFirstShape = Shape {1, 20, 1, 1};
    auto offset = shape_size(convFirstShape) * sizeof(float);
    std::shared_ptr<Node> addFirstConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, addFirstShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> addNodeFirst = std::make_shared<op::v1::Add>(convolutionNodeFirst->output(0), addFirstConstantNode->output(0));

    // -------MAXPOOL----------
    Shape padBeginShape {0, 0};
    Shape padEndShape {0, 0};

    std::shared_ptr<Node> maxPoolingNodeFirst =
        std::make_shared<op::v1::MaxPool>(addNodeFirst->output(0), std::vector<size_t> {2, 2}, padBeginShape, padEndShape, std::vector<size_t> {2, 2},
                                          op::RoundingType::CEIL, op::PadType::EXPLICIT);

    // -------convolution 2----
    auto convSecondShape = Shape {50, 20, 5, 5};
    offset += shape_size(addFirstShape) * sizeof(float);
    std::shared_ptr<Node> convolutionSecondConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, convSecondShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> convolutionNodeSecond =
        std::make_shared<op::v1::Convolution>(maxPoolingNodeFirst->output(0), convolutionSecondConstantNode->output(0), Strides({1, 1}),
                                              CoordinateDiff(padBegin), CoordinateDiff(padEnd), Strides({1, 1}));

    // -------Add 2------------
    auto addSecondShape = Shape {1, 50, 1, 1};
    offset += shape_size(convSecondShape) * sizeof(float);
    std::shared_ptr<Node> addSecondConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, addSecondShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> addNodeSecond = std::make_shared<op::v1::Add>(convolutionNodeSecond->output(0), addSecondConstantNode->output(0));

    // -------MAXPOOL 2--------
    std::shared_ptr<Node> maxPoolingNodeSecond = std::make_shared<op::v1::MaxPool>(addNodeSecond->output(0), Strides {2, 2}, padBeginShape, padEndShape,
                                                                                   Shape {2, 2}, op::RoundingType::CEIL, op::PadType::EXPLICIT);

    // -------Reshape----------
    auto reshapeFirstShape = Shape {2};
    auto reshapeOffset = shape_size(addSecondShape) * sizeof(float) + offset;
    std::shared_ptr<Node> reshapeFirstConstantNode =
        std::make_shared<op::Constant>(element::Type_t::i64, reshapeFirstShape, (weightsPtr->cbuffer().as<uint8_t*>() + reshapeOffset));

    std::shared_ptr<Node> reshapeFirstNode = std::make_shared<op::v1::Reshape>(maxPoolingNodeSecond->output(0), reshapeFirstConstantNode->output(0), true);

    // -------MatMul 1---------
    auto matMulFirstShape = Shape {500, 800};
    offset = shape_size(reshapeFirstShape) * sizeof(int64_t) + reshapeOffset;
    std::shared_ptr<Node> matMulFirstConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, matMulFirstShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> matMulFirstNode = std::make_shared<op::MatMul>(reshapeFirstNode->output(0), matMulFirstConstantNode->output(0), false, true);

    // -------Add 3------------
    auto addThirdShape = Shape {1, 500};
    offset += shape_size(matMulFirstShape) * sizeof(float);
    std::shared_ptr<Node> addThirdConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, addThirdShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> addThirdNode = std::make_shared<op::v1::Add>(matMulFirstNode->output(0), addThirdConstantNode->output(0));

    // -------Relu-------------
    std::shared_ptr<Node> reluNode = std::make_shared<op::Relu>(addThirdNode->output(0));

    // -------Reshape 2--------
    auto reshapeSecondShape = Shape {2};
    std::shared_ptr<Node> reshapeSecondConstantNode =
        std::make_shared<op::Constant>(element::Type_t::i64, reshapeSecondShape, (weightsPtr->cbuffer().as<uint8_t*>() + reshapeOffset));

    std::shared_ptr<Node> reshapeSecondNode = std::make_shared<op::v1::Reshape>(reluNode->output(0), reshapeSecondConstantNode->output(0), true);

    // -------MatMul 2---------
    auto matMulSecondShape = Shape {10, 500};
    offset += shape_size(addThirdShape) * sizeof(float);
    std::shared_ptr<Node> matMulSecondConstantNode =
        std::make_shared<op::Constant>(element::Type_t::f32, matMulSecondShape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> matMulSecondNode = std::make_shared<op::MatMul>(reshapeSecondNode->output(0), matMulSecondConstantNode->output(0), false, true);

    // -------Add 4------------
    auto add4Shape = Shape {1, 10};
    offset += shape_size(matMulSecondShape) * sizeof(float);
    std::shared_ptr<Node> add4ConstantNode = std::make_shared<op::Constant>(element::Type_t::f32, add4Shape, (weightsPtr->cbuffer().as<uint8_t*>() + offset));

    std::shared_ptr<Node> add4Node = std::make_shared<op::v1::Add>(matMulSecondNode->output(0), add4ConstantNode->output(0));

    // -------softMax----------
    std::shared_ptr<Node> softMaxNode = std::make_shared<op::v1::Softmax>(add4Node->output(0), 1);

    // -------ngraph function--
    auto result_full = std::make_shared<op::Result>(softMaxNode->output(0));

    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(result_full, ngraph::ParameterVector {paramNode}, "lenet");

    return fnPtr;
}

/**
 * @brief The entry point for inference engine automatic ngraph function
 * creation sample
 * @file ngraph_function_creation_sample/main.cpp
 * @example ngraph_function_creation_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get Inference Engine version
        // ------------------------------------------------------
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // ------------------------------ Read input
        // -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) {
            throw std::logic_error("No suitable images were found");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;
        // ------------------------------ Get Available Devices
        // ------------------------------------------------------
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        //--------------------------- Step 2. Create network using ngraph function
        //-----------------------------------

        CNNNetwork network(createNgraphFunction());
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Sample supports topologies only with 1 input");
        }

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * Call this before loading the network to the device **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto& i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3], inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }

        if (imagesData.empty()) {
            throw std::logic_error("Valid input images were not found");
        }

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // --------------------------- Prepare output blobs
        // -----------------------------------------------------
        slog::info << "Checking that the outputs are as the sample expects" << slog::endl;
        OutputsDataMap outputInfo(network.getOutputsInfo());
        std::string firstOutputName;

        for (auto& item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("Output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks with a single output");
        }

        DataPtr& output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;

        const SizeVector outputDims = output->getTensorDesc().getDims();
        const int classCount = outputDims[1];

        if (classCount > 10) {
            throw std::logic_error("Incorrect number of output classes for LeNet network");
        }

        if (outputDims.size() != 2) {
            throw std::logic_error("Incorrect output dimensions for LeNet");
        }
        output->setPrecision(Precision::FP32);
        output->setLayout(Layout::NC);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading model to the device
        // ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create infer request
        // -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = exeNetwork.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto& item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Filling input tensor with images. First b channel, then g and r
             * channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixels in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in
                         * bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid * num_channels + ch];
                    }
                }
            }
        }
        inputInfo = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 8. Process output
        // -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr outputBlob = infer_request.GetBlob(firstOutputName);

        /** Validating -nt value **/
        const size_t resultsCnt = outputBlob->size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " << resultsCnt + 1
                       << " and more than 0).\n           Maximal value " << resultsCnt << " will be used.";
            FLAGS_nt = resultsCnt;
        }

        /** Read labels from file (e.x. LeNet.labels) **/
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
            inputFile.close();
        }
        // Prints formatted classification results
        ClassificationResult classificationResult(outputBlob, images, batchSize, FLAGS_nt, labels);
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    slog::info << "This sample is an API example, for performance measurements, "
                  "use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}
