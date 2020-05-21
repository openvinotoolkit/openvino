// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <unordered_set>

#include <gtest/gtest.h>
#include "ie_precision.hpp"
#include <tests_common_func.hpp>
#include <multi-device/multi_device_config.hpp>
#include "low_precision_transformations/transformer.hpp"
#include "common/validation.hpp"
#include "ie_util_internal.hpp"

#include "network_i8.hpp"

/*************************************************
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 * All ref values was obtained from Caffe scoring
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 *************************************************/

TEST_P(ModelTransformationsTest, LPT) {}

ModelParams getModelParams(const std::string modelName) {
std::map<std::string, ModelParams> modelParams = {
    {
        "inception_v3_tf",
        ModelParams(
                "inception_v3_tf",
                "inception_v3/inception_v3_i8.xml",
                "validation_set/299x299/dog.bmp",
                {{157, 10.1683},  // 157 row: 'Blenheim spaniel'
                 { 219, 5.751 },   // 219 row: 'Welsh springer spaniel',
                 { 153, 4.9502 },  // 153 row: 'Japanese spaniel',
                 { 216, 4.79769 }}
        )
    },
        {
            "mobilenet_v2_tf_depthwise",
            ModelParams(
                "mobilenet_v2_tf_depthwise",
                "mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml",
                "validation_set/224x224/dog.bmp",
                {{ 157, 8.63748 },
                 { 219, 6.29954 },
                 { 216, 4.7303 },
                 { 218, 4.69319 },
                 { 220, 3.67249 }},
                {},
                [](const TransformationsParams& transformationsParam, CNNNetworkImplPtr usedNetwork) {
                    if (transformationsParam.transformationsInTestEnabled && transformationsParam.params.updatePrecisions) {
                        const static std::vector<std::pair<std::string, std::string>> fakeQuantizeAndConcolutionItems = {
                            // U8 with shift on activations
                            {"MobilenetV2/Conv/Conv2D/fq_input_0", ""},
                            {"MobilenetV2/expanded_conv/project/Conv2D/fq_input_0", "MobilenetV2/expanded_conv/project/BatchNorm/FusedBatchNormV3/variance/Fused_Add_"},
                            // I8 on activations
                            {"MobilenetV2/expanded_conv_1/expand/Conv2D/fq_input_0", ""},
                            {"MobilenetV2/expanded_conv_1/project/Conv2D/fq_input_0", "MobilenetV2/expanded_conv_1/project/BatchNorm/FusedBatchNormV3/variance/Fused_Add_"},
                            // I8 on activations
                            {"MobilenetV2/expanded_conv_2/add/fq_input_1", ""},
                            {"MobilenetV2/expanded_conv_2/project/Conv2D/fq_input_0", "MobilenetV2/expanded_conv_2/project/BatchNorm/FusedBatchNormV3/variance/Fused_Add_"},
                            // I8 on activations
                            {"MobilenetV2/expanded_conv_3/expand/Conv2D/fq_input_0", ""}
                        };

                        for (const std::pair<std::string, std::string> item : fakeQuantizeAndConcolutionItems) {
                            TestsCommonFunc::checkLayerOuputPrecision(*usedNetwork, item.first, Precision::U8);
                            if (!item.second.empty()) {
                                TestsCommonFunc::checkLayerInputPrecision(*usedNetwork, item.second, Precision::U8, 0);
                            }
                        }
                    }
                })
        },
        {
            "resnet_50_tf",
            ModelParams(
                "resnet_50_tf",
                "resnet_v1_50/resnet_v1_50_i8.xml",
                "validation_set/224x224/dog.bmp",
                {{ 156, 16.1796 },
                 { 218, 11.9186 },
                 { 219, 10.8054 },
                 { 217, 10.1224 },
                 { 152, 9.60148 }},
                {},
                [](const TransformationsParams& transformationsParam, CNNNetworkImplPtr usedNetwork) {
                    if (transformationsParam.transformationsInTestEnabled && transformationsParam.params.updatePrecisions) {
                        const Precision originalPrecision = Precision::FP32;
                        const Precision targetPrecision = Precision::U8;

                        //Eltwise CPU/GPU specific
                        TestsCommonFunc::checkLayerOuputPrecision(*usedNetwork, "resnet_v1_50/block1/unit_1/bottleneck_v1/add/fq_input_0", originalPrecision);
                        TestsCommonFunc::checkLayerOuputPrecision(*usedNetwork, "resnet_v1_50/block1/unit_1/bottleneck_v1/add/fq_input_1", Precision::I8);

                        TestsCommonFunc::checkLayerOuputPrecision(*usedNetwork, "resnet_v1_50/block2/unit_1/bottleneck_v1/add/fq_input_0", originalPrecision);
                        TestsCommonFunc::checkLayerOuputPrecision(*usedNetwork, "resnet_v1_50/block2/unit_1/bottleneck_v1/add/fq_input_1", Precision::I8);
                    }
                })
        },
    };

    const auto it = modelParams.find(modelName);
    if (it == modelParams.end()) {
        THROW_IE_EXCEPTION << "parameters for model '" << modelName << "' were not found";
    }
    return it->second;
}

//0.005f,
INSTANTIATE_TEST_CASE_P(
        smoke_Inception,
        ModelTransformationsTest,
        ::testing::Values(
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, false, createParam(), {}, 3ul),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, true, createParamI8I8(), {}, 0, false),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, true, createParamU8I8(), {}, 0),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, true, createParamU8U8(), {}, 0),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, true, createParamCpu().setQuantizedTensorAlignmentOnActivations(LayerTransformation::QuantizedTensorAlignment::UpdateLevel)),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, false, true, createParamCpu().setQuantizedTensorAlignmentOnActivations(LayerTransformation::QuantizedTensorAlignment::UpdateIntervals)),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 1ul, true, false, createParam()),
                TransformationsParams("CPU", getModelParams("inception_v3_tf"), 2ul, true, false, createParam())
        ),
        TransformationsParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
        smoke_MobileNet,
        ModelTransformationsTest,
        ::testing::Values(
                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, false),
// TODO: eshoguli: fix this issue
//                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, false, true, createParamI8I8()),
//                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, false, true, createParamU8I8()),
//                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, false, true, createParamU8U8(), {}, 2),
//                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, false, true, createParamCpu(), { "464/Pool", "465/Pool" }),
                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 1ul, true),
                TransformationsParams("CPU", getModelParams("mobilenet_v2_tf_depthwise"), 2ul, true)
        ),
        TransformationsParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
        smoke_ResNet,
        ModelTransformationsTest,
        ::testing::Values(
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, false),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, false, true, createParamI8I8(), {
                        // TODO: remove when eltwise validation was added
                        "resnet_v1_50/block1/unit_2/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars",
                        "resnet_v1_50/block2/unit_3/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars"
                }),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, false, true, createParamU8I8(), {
//            // TODO: remove when eltwise validation was added
                        "resnet_v1_50/block1/unit_2/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars",
                        "resnet_v1_50/block2/unit_3/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars"
                }),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, false, true, createParamU8U8(), {
                        // TODO: remove when eltwise validation was added
                        "resnet_v1_50/block1/unit_2/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars",
                        "resnet_v1_50/block2/unit_3/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars"
                }),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, false, true, createParamCpu(), {
                        // TODO: remove when eltwise validation was added
                        "resnet_v1_50/block1/unit_2/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars",
                        "resnet_v1_50/block2/unit_3/bottleneck_v1/act_quant/FakeQuantWithMinMaxVars"
                }),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 1ul, true),
                TransformationsParams("CPU", getModelParams("resnet_50_tf"), 2ul, true)
        ),
        TransformationsParams::getLowPrecisionTransformerSingleLayerTestName);
