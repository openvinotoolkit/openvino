// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <inference_engine/layer_transform.hpp>
#include "gna_plugin/quantization/model_quantizer.hpp"
#include "gna_plugin/quantization/layer_quantizer.hpp"
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class I16QuantisationTest : public GNATest {
 protected:
    LayersQuantizer<QuantI16> lc = LayersQuantizer<QuantI16>(1.0f);

    InferenceEngine::CNNLayerPtr  quantize (InferenceEngine::CNNLayerPtr lp) {
        auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
        transformLayer(newLayer, lc);
        return newLayer;
    };


    void SetUp() override  {
    }

};

template <class T>
T  setWeights(T blob) {
    blob->allocate();
    // actual quantisation algorithm is involved - we need to provide weights that will be quantized with scale factor of 1
    for (auto && w : *blob) {
        w = MAX_VAL_2B_WEIGHT;
    }
    return blob;
}

template <>
TBlob<uint8_t>::Ptr  setWeights(TBlob<uint8_t>::Ptr blob) {
    blob->allocate();
    auto buf = blob->buffer();
    auto ptr = buf.as<float*>();

    for (int i = 0; i != blob->byteSize() / 4; i++) {
        ptr[i] = MAX_VAL_2B_WEIGHT;
    }
    return blob;
}


// TODO: add test for FC weights after quantization
TEST_F(I16QuantisationTest, canQuantizeFCLayer){

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    fc->_out_num = 9;
    fc->_weights = setWeights(make_shared_blob<float>(Precision::FP32, {1, 1}));
    fillWeights(fc->_weights);
    fc->_biases  = make_shared_blob<float>(Precision::FP32, Layout::NC, {1, 1});
    fc->_biases->allocate();
    fillWeights(fc->_biases);

    std::shared_ptr<Data> outData = std::make_shared<Data>("data", SizeVector({1, 1}), Precision::FP32, Layout::NC);
    fc->outData.push_back(outData);
    fc->insData.push_back(outData);


    ASSERT_NO_THROW(quantize(fc));
}

TEST_F(I16QuantisationTest, canQuantizeActivation){

    auto sigmoid = std::make_shared<GenericLayer >(LayerParams{"name", "type", Precision::FP32});
    sigmoid->params["value"] = 2;
    sigmoid->type = "Activation";

    ASSERT_NO_THROW(quantize(sigmoid));
}

TEST_F(I16QuantisationTest, outputAffinePrecisionIs32Bits){

    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(Fc2DOutputModel().data(), Fc2DOutputModel().length()));

    auto weights = make_shared_blob<uint8_t>(Precision::U8, C, {440});
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto affineDataPtr = inputs.begin()->second->getInputData()->inputTo.begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->precision, Precision::I32);
}


TEST_F(I16QuantisationTest, canQuantizeLstmLikeTopology) {
    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(affineToMemoryModel().data(), affineToMemoryModel().length()));

    auto weights = setWeights(make_shared_blob<uint8_t >(Precision::U8, C, {440}));
    //std::fill_n(weights->buffer().as<float*>(), weights->byteSize()/sizeof(float), 0);
    net_reader.SetWeights(weights);

    ASSERT_NO_THROW(q.quantize(net_reader.getNetwork(), 1000));
}

TEST_F(I16QuantisationTest, DISABLED_outputScaleFactorForAffineIsCorrect){

    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(Fc2DOutputModel().data(), Fc2DOutputModel().length()));

    auto weights = make_shared_blob<uint8_t >(Precision::U8, C, {440});
    weights->allocate();
    fillWeights(weights, 100);
    net_reader.SetWeights(weights);

    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto affineLayerPtr = inputs.begin()->second->getInputData()->inputTo.begin()->second;

    auto quantParams = getInjectedData<QuantizedLayerParams>(affineLayerPtr);


    ASSERT_FLOAT_EQ(quantParams->_dst_quant.scale, 100);
    ASSERT_FLOAT_EQ(quantParams->_weights_quant.scale, 100);
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode()
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion_ProfilingEnabled) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode()
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet().profiling_counters();
}

TEST_F(I16QuantisationTest, OnlyAffineWithNanScaleFactorFails) {
    gna()
        .onInferModel(Fc2DOutputModel())
        .withNanScaleFactor()
        .propagate_forward().throws();
}

TEST_F(I16QuantisationTest, OnlyAffineWithInfScaleFactorFails) {
    gna()
        .onInferModel(Fc2DOutputModel())
        .withInfScaleFactor()
        .propagate_forward().throws();
}

TEST_F(I16QuantisationTest, AffineToMemoryWillResultInActivationInsertion) {
    assert_that()
        .onInferModel(affineToMemoryModel())
        .inNotCompactMode()
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemoryWithNoOutputActivationInsertion) {
    assert_that().onInferModel(eltwiseToMemoryModelNoOutput(), [](CNNNetwork & net){
            net.addOutput("Eltwise_8");
        }).inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemory_ActivationInsertion) {
    assert_that().onInferModel(eltwiseToMemoryModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}


TEST_F(I16QuantisationTest, SplitFollowedByActivation_DummyDiagonalAffineInsertion) {
    assert_that().onInferModel(activationAfterSplitModel())
        .inNotCompactMode().gna().propagate_forward().called_with().diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, SplitFollowedByFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {12.0, 12.0, 12.0, 12.0, 12.0,
                                          12.0, 12.0, 12.0, 12.0, 12.0};
    assert_that().onInferModel(FCWithPaddingAfterSplitModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, SliceFollowedByFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0};
    assert_that().onInferModel(FCWithPaddingAfterSliceModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, SliceFollowedByAlignedFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {18.0, 18.0, 18.0, 18.0};
    assert_that().onInferModel(SliceModelWithAlignedOutputs())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, SliceFollowedBy2FCsAnd2EltwisesOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0};
    assert_that().onInferModel(twoFCWithPaddingAfterSliceModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, EltwiseSumm_onlyOneIdentityInsertion) {
    assert_that().onInferModel(eltwiseSummModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet().once();
}


TEST_F(I16QuantisationTest, canDetectLeakyRelu) {
    assert_that().onInferModel(TFLeakyReluModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, MaxPool_followedAfterActivation) {
    assert_that().onInferModel(maxpoolAfterRelu())
        .inNotCompactMode().gna().propagate_forward().called_with()
        .convolution_inserted_into_nnet()
        .And()
        .pwl_inserted_into_nnet()
        .And()
        .max_pooling_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseMull_willInsertTwoIdentities) {
    assert_that().onInferModel(eltwiseMulModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, ConcatPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0};

    assert_that().onInferModel(concatModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, DoubleConcatPropageteForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0};

    assert_that().onInferModel(doubleConcatModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, ScaleShift_Affine_WillResultInIdentityInsertion) {
    assert_that().onInferModel(scaleShiftAffineModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet().once();
}

TEST_F(I16QuantisationTest, ClampFollowedByTanh_ResultInDiagonalInsertion) {
    assert_that().onInferModel(clampFollowedByTanhModel())
        .inNotCompactMode().gna().propagate_forward().called_with().diagonal_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, EltwiseWithMemoryAndActivationInput_ResultInDiagonalInsertion) {
    assert_that().onInferModel(eltwiseWithMemoryAndActivationInputModel())
        .inNotCompactMode().gna().propagate_forward().called_with().diagonal_inserted_into_nnet().once();
}

TEST_F(I16QuantisationTest, AffineWith2AffineOutputs_ResultInOnlyOneIdentityInsertion) {
    // one Identity activation from first FC, and one Identity activation for eltwise
    assert_that().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet().twice();
}

// TODO: this mode not required in rel life scenarios so far
TEST_F(I16QuantisationTest, DISABLED_AffineWithOutputToMemoryAndToAnotherNode_ResultInCopyInsertion) {
    assert_that().onInferModel(affineToMemoryModel()).inNotCompactMode().gna().propagate_forward().
        called_with().copy_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, CropWithoutOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {11.0, 11.0, 11.0, 11.0, 11.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0};

    assert_that().onInferModel(cropWithoutOffsetModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, CropWithAlignedOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {3.0, 3.0, 3.0, 3.0, 3.0,
                                          3.0, 3.0, 3.0, 3.0, 3.0};

    assert_that().onInferModel(cropWithAlignedOffsetModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, CropWithOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {7.0, 7.0, 7.0, 7.0, 7.0,
                                          7.0, 7.0, 7.0, 7.0, 7.0};

    assert_that().onInferModel(cropWithOffsetModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, CropWithMaxOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {1.0, 1.0, 1.0, 1.0, 1.0,
                                          1.0, 1.0, 1.0, 1.0, 1.0};

    assert_that().onInferModel(cropWithMaxOffsetModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, CropWithOffsetAfterFCPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {111.0, 111.0, 111.0, 111.0, 111.0,
                                          111.0, 111.0, 111.0, 111.0, 111.0};

    assert_that().onInferModel(cropWithOffsetExtendedModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(I16QuantisationTest, CopySimpleCasePropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {12.0, 12.0, 12.0, 12.0, 12.0,
                                          12.0, 12.0, 12.0, 12.0, 12.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0,};

    assert_that().onInferModel(copyModel())
    .inNotCompactMode().gna().propagate_forward().onCPU()
    .called_with_input_and_expected_output(input_data, expected_result);
}
