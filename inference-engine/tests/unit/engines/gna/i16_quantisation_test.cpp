// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <inference_engine/layer_transform.hpp>
#include <gna-api-types-xnn.h>
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
    fc->_weights = setWeights(make_shared_blob<float>({ Precision::FP32, {1, 1}, Layout::NC }));
    fillWeights(fc->_weights);
    fc->_biases  = make_shared_blob<float>({ Precision::FP32, {1, 1}, Layout::NC });
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

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {440}, C });
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto affineDataPtr = inputs.begin()->second->getInputData()->getInputTo().begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->getTensorDesc().getPrecision(), Precision::I32);
}


TEST_F(I16QuantisationTest, canQuantizeLstmLikeTopology) {
    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(affineToMemoryModel().data(), affineToMemoryModel().length()));

    auto weights = setWeights(make_shared_blob<uint8_t >({ Precision::U8, {440}, C }));
    //std::fill_n(weights->buffer().as<float*>(), weights->byteSize()/sizeof(float), 0);
    net_reader.SetWeights(weights);

    ASSERT_NO_THROW(q.quantize(net_reader.getNetwork(), 1000));
}

TEST_F(I16QuantisationTest, DISABLED_outputScaleFactorForAffineIsCorrect){

    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(Fc2DOutputModel().data(), Fc2DOutputModel().length()));

    auto weights = make_shared_blob<uint8_t >({ Precision::U8, {440}, C });
    weights->allocate();
    fillWeights(weights, {100});
    net_reader.SetWeights(weights);

    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto affineLayerPtr = inputs.begin()->second->getInputData()->getInputTo().begin()->second;

    auto quantParams = getInjectedData<QuantizedLayerParams>(affineLayerPtr);


    ASSERT_FLOAT_EQ(quantParams->_dst_quant.scale, 100);
    ASSERT_FLOAT_EQ(quantParams->_weights_quant.scale, 100);
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion_ProfilingEnabled) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
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
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemoryWithNoOutputActivationInsertion) {
    assert_that().inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
    .onInferModel(eltwiseToMemoryModelNoOutput(), [](CNNNetwork & net){
            net.addOutput("Eltwise_8");
        }).gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemory_ActivationInsertion) {
    assert_that().onInferModel(eltwiseToMemoryModel()).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}


TEST_F(I16QuantisationTest, SplitFollowedByActivation_DummyDiagonalAffineInsertion) {
    assert_that().onInferModel(activationAfterSplitModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, DISABLED_SliceFollowedBy2FCsAnd2Eltwises_AlignedFilterInsertion) {
    assert_that().onInferModel(twoFCWithPaddingAfterSliceModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().diagonal_inserted_into_nnet();
}

// ToDo requires implementation of aligning filter for concat inputs and improvement of
// qunatization/scaling algorithm for concat
TEST_F(I16QuantisationTest, DISABLED_DoubleConcatPropageteForwardWithSuccess_AlignedFilterInsertion) {
    assert_that().onInferModel(doubleConcatModel()).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode().gna().propagate_forward().called_with().diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseSumm_onlyOneIdentityInsertion) {
    assert_that().onInferModel(eltwiseSummModel()).withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode().gna().propagate_forward().called_with().pwl_inserted_into_nnet().once();
}


TEST_F(I16QuantisationTest, canDetectLeakyRelu) {
    assert_that().onInferModel(TFLeakyReluModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, MaxPool_followedAfterActivation) {
    assert_that().onInferModel(maxpoolAfterRelu())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with()
        .convolution_inserted_into_nnet()
        .And()
        .pwl_inserted_into_nnet()
        .And()
        .max_pooling_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseMull_willInsertTwoIdentities) {
    assert_that().onInferModel(eltwiseMulModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, multiple_inputs_supported) {
    std::string configKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_");
    assert_that().onInferModel(two_inputs_to_affine())
        .inNotCompactMode().withGNAConfig(configKey + std::to_string(0), 1.0f)
        .withGNAConfig(configKey + std::to_string(1), 2.0f).gna().propagate_forward()
        .called_with().pwl_inserted_into_nnet().once();
}
TEST_F(I16QuantisationTest, multiple_inputs_can_handle_individual_scale_factors) {
    std::vector<float> input_data  = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> input2_data = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    std::vector<float> result      = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    std::string configKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_");

    assert_that().onInferModel(two_inputs_to_affine())
        .inNotCompactMode().gna().propagate_forward()
        .called_with().withGNAConfig(configKey + std::to_string(0), 2.0f).And()
        .withGNAConfig(configKey + std::to_string(1), 2.0f).returns().result().filledWith(16384).that().equal_to(result);
}

TEST_F(I16QuantisationTest, DISABLED_multiple_inputs_into_concat_supported) {
    assert_that().onInferModel(two_inputs_to_concat())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).gna().propagate_forward().called_with().pwl_inserted_into_nnet().once();
}

TEST_F(I16QuantisationTest, ScaleShift_Affine_WillResultInIdentityInsertion) {
    assert_that().onInferModel(scaleShiftAffineModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet().once();
}

TEST_F(I16QuantisationTest, ClampFollowedByTanh_ResultInDiagonalInsertion) {
    assert_that().onInferModel(clampFollowedByTanhModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().diagonal_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, EltwiseWithMemoryAndActivationInput_ResultInTwoDiagonalsInsertion) {
    assert_that().onInferModel(eltwiseWithMemoryAndActivationInputModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().diagonal_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, AffineWith2AffineOutputs_ResultInOnlyOneIdentityInsertion) {
    // one Identity activation from first FC, and one Identity activation for eltwise
    assert_that().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwl_inserted_into_nnet().twice();
}

TEST_F(I16QuantisationTest, ScaleShiftWithBroadcast_ResultInDiagonalInsertion) {

    auto & affineWeights = storage<std::vector<uint16_t>>();

    affineWeights = {
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
    };

    assert_that().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).onInferModel(ScaleShift3DModel())
        .withWeigthsPattern({1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f})
        .inNotCompactMode().gna().propagate_forward().called_with().called_with().affine_weights_eq(affineWeights);
}

// TODO: this mode not required in rel life scenarios so far
TEST_F(I16QuantisationTest, DISABLED_AffineWithOutputToMemoryAndToAnotherNode_ResultInCopyInsertion) {
    assert_that().onInferModel(affineToMemoryModel()).inNotCompactMode().gna().propagate_forward().
        called_with().copy_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, DISABLED_permutationOfWeightsBetweenConvAndAffine) {
    auto & affineWeights = storage<std::vector<uint16_t>>();

    // least likely that width and height both are multiple of 7
    auto weigthsPattern = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

    // here weights are transpozed
    save().onInferModel(affineAfterConvNoPermute()).withWeigthsPattern(weigthsPattern)
        .inNotCompactMode().from().propagate_forward().affine_weights_transpozed({128, 61}).to(affineWeights);

    // here weights shouldn't be transposed
    assert_that().onInferModel(affineAfterConvWithPermute()).withWeigthsPattern(weigthsPattern)
        .inNotCompactMode().gna().propagate_forward().called_with().affine_weights_eq(affineWeights);
}

TEST_F(I16QuantisationTest, DISABLED_noPermutationOfWeightsBetweenConvAndAffineIfPermuteLayerWithCorrectArgs) {
    auto & affineWeights = storage<std::vector<uint16_t>>();

    // least likely that width and height both are multiple of 7
    auto weigthsPattern = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

    save().onInferModel(affineAfterConvWithPermute()).withWeigthsPattern(weigthsPattern)
        .inNotCompactMode().from().propagate_forward().affine_weights().to(affineWeights);

    assert_that().onInferModel(affineAfterConvNoPermute()).withWeigthsPattern(weigthsPattern)
        .inNotCompactMode().gna().propagate_forward().called_with().affine_weights_transposed(affineWeights, {128, 61});
}

TEST_F(I16QuantisationTest, fp16tofp32_on_fullyConnected_model) {
    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(FCOnlyModelFP16().data(), FCOnlyModelFP16().length()));

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {220}, Layout::C });
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    q.quantize(net_reader.getNetwork(), 1000);
}


TEST_F(I16QuantisationTest, MultipleActivationsAfterAffineWithIdentityActivation_MultipleDiagonalLayersWithActivaitons) {
    // identiy came from automatic insertion due to
    assert_that().onInferModel(AffineWithReluSigmoidAndIdentity())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwls_inserted_into_nnet({kActSigmoid, kActRelu, kActIdentity, kActIdentity});
}

TEST_F(I16QuantisationTest, MultipleActivationsAfterAffine_ResultInMultipleDiagonalLayersWithActivaitons) {
    // extra identity inserted for affine
    assert_that().onInferModel(AffineWithReluSigmoid())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with()
         // 1 diag for second activation, 1 for eltwise
        .pwls_inserted_into_nnet({kActRelu, kActSigmoid}).diagonal_inserted_into_nnet().times(3);
}

// TODO: build a regression test on top of it using real quantisation accuracy checking
TEST_F(I16QuantisationTest, ConcatWithConstInputPropagatedForward) {
    assert_that().onInferModel(concatModelWithConstLayer())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwls_inserted_into_nnet({kActIdentity});
}

TEST_F(I16QuantisationTest, LSTMCell_quantize) {
    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(LSTMCellOnlyModel().data(), LSTMCellOnlyModel().length()));

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {33664}, C });
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    ASSERT_NO_THROW(q.quantize(net_reader.getNetwork(), 1000));
}

TEST_F(I16QuantisationTest, LSTMCell_unaligned_quantize) {
    ModelQuantizer<QuantI16> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(LSTMCellOnlyModelUnaligned().data(), LSTMCellOnlyModelUnaligned().length()));

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {3480}, C });
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    ASSERT_NO_THROW(q.quantize(net_reader.getNetwork(), 1000));
}

TEST_F(I16QuantisationTest, EltwisetWithConstInputPropagatedForward) {
    assert_that().onInferModel(eltwiseSumModelWithConstLayer())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, PowerWithScaleFactorPropagateForward) {
    assert_that().onInferModel(PowerWithScaleFactor1())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna().propagate_forward().called_with().pwls_inserted_into_nnet({kActIdentity}).And().diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, ConcatWithDifferentInputScaleFactorsPropagateForward) {
    assert_that().onInferModel(ConcatWithDiffScaleFactor())
            .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
            .gna().propagate_forward().called_with().pwls_inserted_into_nnet({kActIdentity});
}
