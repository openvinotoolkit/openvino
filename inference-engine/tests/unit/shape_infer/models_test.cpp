// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/cnn_network_impl.hpp>
#include <inference_engine/shape_infer/ie_reshaper.hpp>
#include "details/ie_cnn_network_tools.h"
#include <cpp/ie_cnn_net_reader.h>
#include <graph_tools.hpp>
#include <test_model_path.hpp>
#include <xml_helper.hpp>
#include <file_utils.h>
#include "built_in_shape_infer_general_test.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;

class GeneralShapeInferModelsTests
        : public BuiltInShapeInferTestWithParam<std::tuple<InOutShapes, ModelPath, CanInfer>> {
protected:
    void SetUp() override {
        BuiltInShapeInferCommon::SetUp();
        auto params = GetParam();
        inOutShapes = std::get<0>(params);
        modelsPath = std::get<1>(params);
        canInfer = std::get<2>(params);
    }

protected:
    testing::InOutShapes inOutShapes;
    bool canInfer{};
    std::string modelsPath;
};

class OriginalShapeInferModelsTests
        : public BuiltInShapeInferTestWithParam<std::string> {
protected:
    void SetUp() override {
        BuiltInShapeInferCommon::SetUp();
        modelsPath = GetParam();
    }

    std::map<std::string, SizeVector> getShapes(const std::vector<CNNLayerPtr>& allLayers) {
        std::map<std::string, SizeVector> shapes;
        for (const auto& layer:allLayers) {
            for (const auto& data:layer->outData) {
                shapes[data->getName()] = data->getTensorDesc().getDims();
            }
        }
        return shapes;
    }

    void compare(const std::vector<CNNLayerPtr>& allLayers, std::map<std::string, SizeVector>& oldShapes) {
        for (const auto& layer:allLayers) {
            for (const auto& data:layer->outData) {
                ASSERT_EQ(oldShapes[data->getName()], data->getTensorDesc().getDims())
                                            << "Shapes don't match:\n Data Name: " << data->getName() << "\n Layer Name: "
                                            << layer->name << "\n Layer Type: " << layer->type;
            }
        }
    }

protected:
    std::string modelsPath;
};

TEST_P(GeneralShapeInferModelsTests, reshape) {
    CNNNetReader reader;
    auto modelPath = ModelsPath() + kPathSeparator + modelsPath;
    reader.ReadNetwork(modelPath);
    reader.ReadWeights(FileUtils::fileNameNoExt(modelPath) + ".bin");
    auto iCnnNetwork = reader.getNetwork();

    auto reshaper = std::make_shared<Reshaper>(iCnnNetwork);
    auto inputShapes = setInputShapes(iCnnNetwork, inOutShapes.inDims);

    if (canInfer) {
        reshaper->run(inputShapes);
        checkNetworkInOut(iCnnNetwork, inOutShapes);
    } else {
        ASSERT_THROW(reshaper->run(inputShapes), InferenceEngine::details::InferenceEngineException);
    }
}

TEST_P(OriginalShapeInferModelsTests, reshape) {
    CNNNetReader reader;
    auto modelPath = ModelsPath() + kPathSeparator + modelsPath;
    reader.ReadNetwork(modelPath);
    reader.ReadWeights(FileUtils::fileNameNoExt(modelPath) + ".bin");
    auto network = reader.getNetwork();
    auto allLayers = CNNNetSortTopologically(network);
    auto oldShapes = getShapes(allLayers);
    CNNNetwork cppNet = reader.getNetwork();
    auto inputShapes = cppNet.getInputShapes();
    auto reshaper = std::make_shared<Reshaper>(network);
    try {
        reshaper->run(inputShapes);
    } catch (const InferenceEngineException& e) {
        FAIL() << e.what();
    }
    compare(allLayers, oldShapes);
}

class SimpleBatchShapeInferModelsTests : public OriginalShapeInferModelsTests {
};

TEST_P(SimpleBatchShapeInferModelsTests, simpleBatch) {
    const int batch = 777;
    CNNNetReader reader;
    auto modelPath = ModelsPath() + kPathSeparator + modelsPath;
    reader.ReadNetwork(modelPath);
    reader.ReadWeights(FileUtils::fileNameNoExt(modelPath) + ".bin");
    auto network = reader.getNetwork();
    auto allLayers = CNNNetSortTopologically(network);
    auto oldShapes = getShapes(allLayers);
    CNNNetwork cppNet = reader.getNetwork();
    auto inputShapes = cppNet.getInputShapes();
    for (auto& shape:inputShapes) {
        auto dims = shape.second;
        auto name = shape.first;
        dims[0] = batch;
        inputShapes[name] = dims;
    }
    auto reshaper = std::make_shared<Reshaper>(network);
    try {
        reshaper->run(inputShapes);
    } catch (const InferenceEngineException& e) {
        FAIL() << e.what();
    }

    for (auto& shape :oldShapes) {
        shape.second[0] = batch;
    }
    compare(allLayers, oldShapes);
}

INSTANTIATE_TEST_CASE_P(
        NewShapesForModels, GeneralShapeInferModelsTests,
        ::testing::Values(
                ::testing::make_tuple(InOutShapes({{{3, 3, 227, 227}},
                                                   {{3, 4, 109, 109}, {3, 2, 109, 109}}}),
                                      ModelPath("mtcnn/PNet_fp16.xml"), CanInfer(true)),
                ::testing::make_tuple(InOutShapes({{{1, 3, 1000, 1000}},
                                                   {{}}}),
                                      ModelPath("alexnet/bvlc_alexnet_fp16.xml"), CanInfer(false)),
                ::testing::make_tuple(InOutShapes({{{1, 3,  1000, 1000}},
                                                   {{1, 21, 1000, 1000}}}),
                                      ModelPath("fcn/fcn8s-heavy-pascal_fp16.xml"), CanInfer(true)),
                ::testing::make_tuple(InOutShapes({{{1, 3, 1000, 1000}},
                                                   {{}}}),
                                      ModelPath("googlenet/bvlc_googlenet_fp16.xml"), CanInfer(false)),
                ::testing::make_tuple(InOutShapes({{{7, 3, 300, 300}},
                                                   {{1, 1, 700, 7}}}),
                                      ModelPath("MobileNet-SSD/MobileNet-SSD_fp16.xml"), CanInfer(true)),
                ::testing::make_tuple(InOutShapes({{{7, 3, 300,  300}},
                                                   {{1, 1, 1400, 7}}}),
                                      ModelPath("SSD_300/ssd_300_fp16.xml"), CanInfer(true)),
                ::testing::make_tuple(InOutShapes({{{2, 3, 500, 500}},
                                                   {{1, 1, 400, 7}}}),
                                      ModelPath("SSD_512/ssd_512_fp16.xml"), CanInfer(true)),
                ::testing::make_tuple(InOutShapes({{{1, 3, 24, 94}, {88, 1}},
                                                   {{}}}),
                                      ModelPath("lprnet/LPRNet_new_fp32.xml"),
                                      CanInfer(false))  // can handle invalid IR without segfault
//                ::testing::make_tuple(InOutShapes({{{1, 3,    400, 400}},
//                                                   {{1, 1001, 2,   2}}}),
//                                      ModelPath("GoogleNet-V3-TF/inception.xml"), CanInfer(true)) // TODO: is data axis="0" dim="1,1001" num_axes="-1" not enough for Reshape. need more attributes?
        )
);

static std::vector<std::string> advancedBatchModels =
        {"PC-Detect_0026/PVANET_fp32.xml",
         "Perc_ZF_fasterRCNN/Perc_ZF_fasterRCNN_fp32.xml", // TODO: try to infer batch
         "SSD_512/ssd_512_fp16.xml",
         "SSD_GoogleNet_v2/SSD_GoogleNet_v2_fp32.xml",
         "ssd_weights_performance/ssd_fp32.xml",
         "ssd_weights_performance/ssd_slow_fp32.xml",
         "squeezenet_ssd/squeezenet_ssd_fp32.xml",
         "SSD_VGG/VGG_ILSVRC2016_SSD_300x300_deploy_fp32.xml",
         "squeezenet_ssd_224_akhila/squeezenet_ssd_224_akhila_fp32.xml",
         "lprnet-48/LPRNet_48_fp32.xml",
         "lprnet-43/LPRNet_43_fp32.xml",
         "lprnet-dla_25k/LPRNet_DLA_25k_fp32.xml",
         "lprnet-70/LPRNet_fp32.xml",
         "rm_lstm4f/rm_lstm4f_fp32.xml", // Changing shape (batch) is not supported now with hiding Memory layers from users
         "fake-pvanet/PVANET_fp32.xml",
         "SSD_300/ssd_300_fp16.xml",
         "chuanqi305_squeezenet_ssd/squeezenet_ssd_fp32.xml",
         "face-detection-retail-0004/FP16/face-detection-retail-0004.xml",
         // "icvnet1.3-ssd/ICV1.3_320_orig_hyper_6_coco_iter_400000.pb.xml", // TODO: get correct IR, and Const is not Input and can't change batch for it
        };

static std::vector<std::string> simpleBatchModels =
        {"cyberlink/style-frozen.xml",
         "cyberlink/refine-frozen.xml",
         "cyberlink/style-frozen-5-304-908.xml",
         "cyberlink/refine-frozen-3-1208-404.xml",
         "icv-hpe/icv-hpe-pre-poc-0001-201-102.xml",
         "shape_infer/tf_alexnet.xml",
         "shape_infer/tf_mobilenet_batch5.xml",
         "shape_infer/tf_mobilenet_orig.xml",
         "fcn/fcn8s-heavy-pascal_fp16.xml",
         "alexnet/bvlc_alexnet_fp16.xml",
         "GoogleNet-V1-TF/inceptionV1-501-402.xml",
         "mtcnn/PNet_fp16.xml",
         "conv_conv_med/conv_conv_med_fp16.xml",
         "densenet-121/DENSENET_121_32.xml",
         "MobileNet/mobilenet_fp16.xml",
         "googlenet/bvlc_googlenet_fp16.xml",
         "resnet-18/ResNet-18_fp32.xml",
         "ResNet-50/ResNet-50_fp32.xml",
         "ResNet-101/ResNet-101_fp32.xml",
         "ResNet-152/ResNet-152_fp32.xml",
         "vgg/VGG_ILSVRC_16_layers_fp32.xml",
         "vgg-16/vgg16_tf_fp32.xml",
         "vgg-19/VGG_ILSVRC_19_layers_fp32.xml",
         "yolo-full/yolo-full_fp32.xml",
         "yolo-tiny-internal/yolo-tiny_fp32.xml",
         "yolo_v2/YoloV2_fp32.xml",
         "yolo_tiny_v1/tiny_yolo_v1_fp32.xml",
         "pvanet-reid/PVANET_Reid_fp32.xml",
         "tsrnet/tsrnet_fp32.xml",
         "pvanet-reid-simplifier/reid_pva_fp32.xml",
         "eu-speed-limit-net/speednet_eu_fp32.xml",
         "dlia_sandbox/conv_multi_out/conv_multi_out_fp32.xml",
         "dlia_sandbox/conv_conv_fc_med/conv_conv_fc_med_fp32.xml",
         "dlia_sandbox/conv_pool_relu_fc_med/conv_pool_relu_fc_med_fp32.xml",
         "dlia_sandbox/eltwise/eltwise_fp32.xml",
         "dlia_sandbox/concat/concat_straight_order_fp32.xml",
         "dlia_sandbox/concat/concat_revert_order_fp32.xml",
         "SqueezeNet_v1.1/SqueezeNet_v1.1_modified_fp32.xml",
         "SqueezeNet_v1.1/SqueezeNet_v1.1_fp32.xml",
         "squeezenet_ssd/snssd_a_1_2_backbone_fp32.xml",
         "dlia_sandbox/reid_stuff_test/ReID_stuff_test_fp32.xml",
         "dlia_sandbox/multi_fc_output_test/Multi_FC_output_test_fp32.xml",
         "dlia_sandbox/relu_test/relu_test_fp32.xml",
         "MobileNet/mobilenet_fp32.xml",
         "dlia_sandbox/dilated_convolution/dilated_convolution_fp32.xml",
         "dlia_sandbox/scale_tests/scale_test_0_fp32.xml",
         "dlia_sandbox/scale_tests/scale_test_1_fp32.xml",
         "dlia_sandbox/scale_tests/scale_test_2_fp32.xml",
         "dlia_sandbox/scale_tests/scale_test_3_fp32.xml",
         "dlia_sandbox/two_fc_out_from_conv/two_fc_out_from_conv_fp32.xml",
         "dlia_sandbox/pooling_with_pads/pooling_with_pads_fp32.xml",
         "dlia_sandbox/fc_branching_test/fc_branching_test_fp32.xml",
         "SqueezeNet_v1.1_pool_kernel_13/SqueezeNet_v1.1_pool_kernel_13_fp32.xml",
         "dlia_sandbox/output_transform_with_slicing_test/output_transform_with_slicing_test_fp32.xml",
        };

INSTANTIATE_TEST_CASE_P(
        CanCalculateOriginalShapesSimple, OriginalShapeInferModelsTests, ::testing::ValuesIn(simpleBatchModels));

INSTANTIATE_TEST_CASE_P(
        CanCalculateOriginalShapesAdvanced, OriginalShapeInferModelsTests, ::testing::ValuesIn(advancedBatchModels));

INSTANTIATE_TEST_CASE_P(
        SimpleBatch, SimpleBatchShapeInferModelsTests, ::testing::ValuesIn(simpleBatchModels));
