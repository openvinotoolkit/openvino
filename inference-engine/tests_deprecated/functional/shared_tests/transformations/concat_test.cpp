// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/concat.hpp"
#include "low_precision_transformations/eltwise.hpp"
#include "common_test_utils/common_utils.hpp"

ConcatTestModel::ConcatTestModel(
    const bool signedIntervals,
    const bool symmetricInterval,
    const bool multiChannel,
    const std::vector<size_t>& constInputDimentions) :
    signedIntervals(signedIntervals),
    symmetricInterval(symmetricInterval),
    multiChannel(multiChannel),
    constInputDimentions(constInputDimentions) {}

std::string ConcatTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
//    ASSERT_EQ(2, p.inputDimensions.size());
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t axis = 1; // should be passed in 'p' argument

    std::vector<size_t> concat_out_dims = p.inputDimensions[0];
    concat_out_dims[axis] += p.inputDimensions[1][axis];

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {{"levels", "256"}};
    std::map<std::string, std::string> concat_params = {{"axis", "1"}};
    std::map<std::string, std::string> power_params = { {"power", "1"}, {"scale", "1"}, {"shift", "0"} };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "10,10"}, {"1,1", "11,16"}, // Inputs to FakeQuantize
        {"2,2", "10,11"}, {"3,3", "10,12"}, {"4,4", "10,13"}, {"5,5", "10,14"}, // Const layers
        {"6,6", "11,17"}, {"7,7", "11,18"}, {"8,8", "11,19"}, {"9,9", "11,20"}, // Const layers
        {"10,15", "12,22"}, {"11,21", "12,23"} // FakeQuantize to Concat
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Concat_transformations_", p.inputDimensions[0], p._network_precision)
        .addInputLayer(p._network_precision, p.inputDimensions[1])
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {constInputDimentions}}, type_size, 0)
        .addLayer(
            "FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            {{p.inputDimensions[0], constInputDimentions, constInputDimentions, constInputDimentions, constInputDimentions}, {{p.inputDimensions[0]}}},
            "fakeQuantize1")
        .addLayer(
            "FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            {{p.inputDimensions[1], constInputDimentions, constInputDimentions, constInputDimentions, constInputDimentions}, {{p.inputDimensions[1]}}},
            "fakeQuantize2")
        .addLayer("Concat", p._network_precision, &concat_params, { {p.inputDimensions[0], p.inputDimensions[1]}, { concat_out_dims }}, "concat")
        .finish(&edges);
}

std::string ConcatTestModel::getName() const {
    return std::string("ConcatTestModel") +
        (signedIntervals ? "_Signed" : "_Unsigned") +
        (symmetricInterval ? "_Symmetric" : "_Asymmetric") +
        (multiChannel ? "_MultiChannel" : "_OneChannel") +
        (constInputDimentions.size() == 1ul ? "" : ("_const" + std::to_string(constInputDimentions.size()) + "D"));
}

bool ConcatTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    // TODO: remove when updatePrecisions is configurable
    params.updatePrecisions = true;

    LowPrecisionTransformations transformations = getLowPrecisionTransformations(params);

    if (!multiChannel) {
        // avoid ConcatMultiChannelsTransformation
        transformations = transformations.
            removeBranchSpecificTransformations("Concat").
            addBranchSpecific<ConcatTransformation>(params, "Concat");
    }

    LowPrecisionTransformer transformer(transformations);
    transformer.transform(network);

    const CNNLayerPtr concatLayer = CommonTestUtils::getLayerByName(network, "concat");
    if (concatLayer == nullptr) {
        THROW_IE_EXCEPTION << "concat layer was not found";
    }

    const std::vector<size_t> dims = concatLayer->outData[0]->getDims();
    if (dims.size() == 4ul) {
        const CNNLayerPtr fakeQuantizeLayer1 = CommonTestUtils::getLayerByName(network, "fakeQuantize1");
        QuantizeLayer* fakeQuantize1 = dynamic_cast<QuantizeLayer*>(fakeQuantizeLayer1.get());
        if (fakeQuantize1 == nullptr) {
            THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer1->name;
        }
        if (fakeQuantize1->levels == 0) {
            //
        }

        const CNNLayerPtr fakeQuantizeLayer2 = CommonTestUtils::getLayerByName(network, "fakeQuantize2");
        QuantizeLayer* fakeQuantize2 = dynamic_cast<QuantizeLayer*>(fakeQuantizeLayer2.get());
        if (fakeQuantize2 == nullptr) {
            THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer2->name;
        }
        if (fakeQuantize2->levels == 0) {
            //
        }
    } else if (dims.size() == 2ul) {
        if (concatLayer->outData[0]->getInputTo().size() != 0ul) {
            THROW_IE_EXCEPTION << "2D is not supported";
        }
    }
    return true;
}

void ConcatTestModel::resetTransformation(CNNNetwork& network) const {
    const float intervalsCoefficient = 0.5f;
    if (signedIntervals) {
        const float symmetricCoefficient = symmetricInterval ? 1.f : 0.5f;
        fillData(getLayer(network, "Const2"), (-128.f / 20.0) * symmetricCoefficient * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const3"), (127.f / 20.0) * symmetricCoefficient * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const4"), (-128.f / 20.0) * symmetricCoefficient * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const5"), (127.f / 20.0) * symmetricCoefficient * intervalsCoefficient, "custom");

        fillData(getLayer(network, "Const6"), (-128.f / 20.0) * symmetricCoefficient, "custom");
        fillData(getLayer(network, "Const7"), 127.f / 20.0, "custom");
        fillData(getLayer(network, "Const8"), (-128.f / 20.0) * symmetricCoefficient, "custom");
        fillData(getLayer(network, "Const9"), 127.f / 20.0, "custom");

    } else {
        const float shift = symmetricInterval ? 0.f : (255.f / 20.0) / 4.f;
        fillData(getLayer(network, "Const2"), (0.0 + shift) * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const3"), (255.f / 20.0) * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const4"), (0.0 + shift) * intervalsCoefficient, "custom");
        fillData(getLayer(network, "Const5"), (255.f / 20.0) * intervalsCoefficient, "custom");

        fillData(getLayer(network, "Const6"), 0.f, "custom");
        fillData(getLayer(network, "Const7"), 255.f / 20.0, "custom");
        fillData(getLayer(network, "Const8"), 0.f, "custom");
        fillData(getLayer(network, "Const9"), 255.f / 20.0, "custom");
    }
}

float ConcatTestModel::getThreshold(const std::string& device_name, const Precision precision, LayerTransformation::Params& params) const {
    if (device_name == "CPU") {
        if (params.updatePrecisions) {
            // FakeQuantize intervals are rounded in INT8 and as result threshold is increased
            return 0.0250001f;
        }
    }

    if (device_name == "GPU") {
        if (precision == Precision::FP32) {
            return 0.00200001f;
        } else {
            return 0.00062f;
        }
    }

    return SingleLayerTestModel::getThreshold(device_name, precision, params);
}
