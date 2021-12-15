// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>
#include <gtest/gtest.h>
#include <ie_data.h>
#include "ie_precision.hpp"
#include <legacy/ie_layers_internal.hpp>

using namespace std;
using InferenceEngine::X_AXIS;
using InferenceEngine::Y_AXIS;
using namespace InferenceEngine;

const std::string defaultLayerName = "layer";
const std::string defaultLayerType = "unknown";
InferenceEngine::Precision defaultPrecision{InferenceEngine::Precision::FP32};


class LayersTests : public ::testing::Test {
public:
    static InferenceEngine::LayerParams getParamsForLayer(std::string name, std::string type,
                                                          InferenceEngine::Precision precision) {
        InferenceEngine::LayerParams params = {};
        params.name = name;
        params.type = type;
        params.precision = precision;
        return params;
    }

    static InferenceEngine::LayerParams getDefaultParamsForLayer() {
        return getParamsForLayer(defaultLayerName, defaultLayerType, defaultPrecision);
    }

    template<class T>
    bool checkCreateLayer() {
        T layer(getDefaultParamsForLayer());
        return layer.name == defaultLayerName;
    }
};

TEST_F(LayersTests, canCreateLayersWithDefaultParams) {
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::CNNLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ConvolutionLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::DeconvolutionLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::PoolingLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::PowerLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::FullyConnectedLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ConcatLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::SplitLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::NormLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::SoftMaxLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::GRNLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ReLULayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::EltwiseLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::CropLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ScaleShiftLayer>());

}

TEST_F(LayersTests, throwsOnExpiredDataPtr) {
    InferenceEngine::CNNLayer layer(getDefaultParamsForLayer());
    InferenceEngine::DataPtr dataPtr(
        new InferenceEngine::Data("data", TensorDesc(InferenceEngine::Precision::FP32, InferenceEngine::NCHW)));
    layer.insData.resize(1);
    layer.insData[0] = dataPtr;
    dataPtr.reset();
    ASSERT_THROW(layer.input(), InferenceEngine::Exception);
}

template<class T>
void ASSERT_CNN_LAYER_DEFAULT(T& l) {
    ASSERT_STREQ(l.params["some"].c_str(), "some");
}

template<>
void ASSERT_CNN_LAYER_DEFAULT(ConvolutionLayer& l) {
    ASSERT_EQ(l._kernel[X_AXIS], 1);
    ASSERT_EQ(l._kernel_x, 1);
    ASSERT_EQ(l._out_depth, 3);
    ASSERT_EQ(l._group, 4);
    ASSERT_EQ(l._pads_end[X_AXIS], 5);
    ASSERT_EQ(l._stride[X_AXIS], 6);
    ASSERT_EQ(l._stride_x, 6);
    ASSERT_EQ(l._dilation[X_AXIS], 7);
    ASSERT_EQ(l._dilation_x, 7);
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(static_cast<CNNLayer&>(l)));
}

template<>
void ASSERT_CNN_LAYER_DEFAULT(DeconvolutionLayer& l) {
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(static_cast<ConvolutionLayer&>(l)));
}

template<>
void ASSERT_CNN_LAYER_DEFAULT(PoolingLayer& l) {
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(static_cast<CNNLayer&>(l)));
}

template<class T>
void checkCopyLayerWithKernel(T& layer_1) {
    T layer_2(LayersTests::getDefaultParamsForLayer());

    layer_2 = layer_1;
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_2));

    T layer_3(layer_1);
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_3));

    T layer_4(LayersTests::getDefaultParamsForLayer());
    //use copy to temporary object then move it using asignment operator
    {
        T layerx(layer_1);
        layer_4 = std::move(layerx);
    }

    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_4));

    //use copy to temporary object then move it using move ctor
    T layerx(layer_1);
    T layer_5 = std::move(layerx);
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_5));


    layer_1._kernel[X_AXIS] = 4;
    ASSERT_EQ(layer_1._kernel_x, 4);
    layer_1._kernel_x = 40;
    ASSERT_EQ(layer_1._kernel[X_AXIS], 40);

    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_2));
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_3));
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_4));
    ASSERT_NO_FATAL_FAILURE(ASSERT_CNN_LAYER_DEFAULT(layer_5));
}


TEST_F(LayersTests, copyConvolution) {
    ConvolutionLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(X_AXIS, 1);
    layer_1._padding.insert(X_AXIS, 2);
    layer_1._out_depth = 3;
    layer_1._group = 4;
    layer_1._pads_end.insert(X_AXIS, 5);
    layer_1._stride.insert(X_AXIS, 6);
    layer_1._dilation.insert(X_AXIS, 7);
    layer_1.params["some"] = "some";
    checkCopyLayerWithKernel(layer_1);
}

TEST_F(LayersTests, copyDeconvolution) {
    ConvolutionLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(X_AXIS, 1);
    layer_1._padding.insert(X_AXIS, 2);
    layer_1._out_depth = 3;
    layer_1._group = 4;
    layer_1._pads_end.insert(X_AXIS, 5);
    layer_1._stride.insert(X_AXIS, 6);
    layer_1._dilation.insert(X_AXIS, 7);
    layer_1.params["some"] = "some";
    checkCopyLayerWithKernel(layer_1);
}

TEST_F(LayersTests, copyPooling) {
    PoolingLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(X_AXIS, 1);
    layer_1._padding.insert(X_AXIS, 2);
    layer_1._type = PoolingLayer::MAX;
    layer_1._exclude_pad = true;
    layer_1._pads_end.insert(X_AXIS, 5);
    layer_1._stride.insert(X_AXIS, 6);
    layer_1.params["some"] = "some";
    checkCopyLayerWithKernel(layer_1);
}

TEST_F(LayersTests, canNotInserOutOfBounds) {

    PoolingLayer layer_1(getDefaultParamsForLayer());
    ASSERT_ANY_THROW(layer_1._kernel.insert(MAX_DIMS_NUMBER, 5));
    ASSERT_ANY_THROW(layer_1._kernel.insert(-1, 5));
}

TEST_F(LayersTests, canInsertIntoExistedAxis) {
    PoolingLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(Z_AXIS, 5);
    ASSERT_EQ(layer_1._kernel[Z_AXIS], 5);
    // for backward compatibility with IRv2 initial size is 2
    ASSERT_EQ(layer_1._kernel.size(), 3);

    layer_1._kernel.insert(Z_AXIS, 6);
    ASSERT_EQ(layer_1._kernel[Z_AXIS], 6);
    ASSERT_EQ(layer_1._kernel.size(), 3);

    layer_1._kernel.insert(Z_AXIS, 0);
    ASSERT_EQ(layer_1._kernel[Z_AXIS], 0);
    ASSERT_EQ(layer_1._kernel.size(), 3);

    layer_1._kernel.insert(Z_AXIS, 6);
    ASSERT_EQ(layer_1._kernel[Z_AXIS], 6);
    ASSERT_EQ(layer_1._kernel.size(), 3);
}

TEST_F(LayersTests, canRemoveProperty) {
    PoolingLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(Z_AXIS, 5);
    layer_1._kernel.remove(X_AXIS);
    layer_1._kernel.remove(Y_AXIS);
    ASSERT_EQ(layer_1._kernel.size(), 1);
    layer_1._kernel.remove(Z_AXIS);
    ASSERT_EQ(layer_1._kernel.size(), 0);
}

TEST_F(LayersTests, cannotRemovePropertyOutOfRange) {

    PoolingLayer layer_1(getDefaultParamsForLayer());
    layer_1._kernel.insert(Z_AXIS, 5);
    ASSERT_NO_THROW(layer_1._kernel.remove(MAX_DIMS_NUMBER + 1000000));
    ASSERT_NO_THROW(layer_1._kernel.remove(-1));
    ASSERT_EQ(layer_1._kernel.size(), 3);
}

TEST_F(LayersTests, convIRv2BackwardCompatibility) {
    ConvolutionLayer conv(getDefaultParamsForLayer());
    ASSERT_NO_THROW(conv._kernel[X_AXIS]);
    ASSERT_NO_THROW(conv._kernel[Y_AXIS]);
    ASSERT_NO_THROW(conv._padding[X_AXIS]);
    ASSERT_NO_THROW(conv._padding[Y_AXIS]);
    ASSERT_NO_THROW(conv._stride[X_AXIS]);
    ASSERT_NO_THROW(conv._stride[Y_AXIS]);
    ASSERT_NO_THROW(conv._dilation[X_AXIS]);
    ASSERT_NO_THROW(conv._dilation[Y_AXIS]);

    conv._kernel_x = 9u;
    ASSERT_EQ(conv._kernel[X_AXIS], 9u);
    conv._kernel_y = 2u;
    ASSERT_EQ(conv._kernel[Y_AXIS], 2u);
    conv._padding_x = 3u;
    ASSERT_EQ(conv._padding[X_AXIS], 3u);
    conv._padding_y = 4u;
    ASSERT_EQ(conv._padding[Y_AXIS], 4u);
    conv._stride_x = 5u;
    ASSERT_EQ(conv._stride[X_AXIS], 5u);
    conv._stride_y = 6u;
    ASSERT_EQ(conv._stride[Y_AXIS], 6u);
    conv._dilation_x = 7u;
    ASSERT_EQ(conv._dilation[X_AXIS], 7u);
    conv._dilation_y = 8u;
    ASSERT_EQ(conv._dilation[Y_AXIS], 8u);

    conv._kernel[X_AXIS] = 8u;
    ASSERT_EQ(conv._kernel_x, 8u);
    conv._kernel[Y_AXIS] = 7u;
    ASSERT_EQ(conv._kernel_y, 7u);
    conv._padding[X_AXIS] = 6u;
    ASSERT_EQ(conv._padding_x, 6u);
    conv._padding[Y_AXIS] = 5u;
    ASSERT_EQ(conv._padding_y, 5u);
    conv._stride[X_AXIS] = 4u;
    ASSERT_EQ(conv._stride_x, 4u);
    conv._stride[Y_AXIS] = 3u;
    ASSERT_EQ(conv._stride_y, 3u);
    conv._dilation[X_AXIS] = 2u;
    ASSERT_EQ(conv._dilation_x, 2u);
    conv._dilation[Y_AXIS] = 9u;
    ASSERT_EQ(conv._dilation_y, 9u);
}

TEST_F(LayersTests, deconvIRv2BackwardCompatibility) {
    DeconvolutionLayer deconv(getDefaultParamsForLayer());
    ASSERT_NO_THROW(deconv._kernel[X_AXIS]);
    ASSERT_NO_THROW(deconv._kernel[Y_AXIS]);
    ASSERT_NO_THROW(deconv._padding[X_AXIS]);
    ASSERT_NO_THROW(deconv._padding[Y_AXIS]);
    ASSERT_NO_THROW(deconv._stride[X_AXIS]);
    ASSERT_NO_THROW(deconv._stride[Y_AXIS]);
    ASSERT_NO_THROW(deconv._dilation[X_AXIS]);
    ASSERT_NO_THROW(deconv._dilation[Y_AXIS]);

    deconv._kernel_x = 9u;
    ASSERT_EQ(deconv._kernel[X_AXIS], 9u);
    deconv._kernel_y = 2u;
    ASSERT_EQ(deconv._kernel[Y_AXIS], 2u);
    deconv._padding_x = 3u;
    ASSERT_EQ(deconv._padding[X_AXIS], 3u);
    deconv._padding_y = 4u;
    ASSERT_EQ(deconv._padding[Y_AXIS], 4u);
    deconv._stride_x = 5u;
    ASSERT_EQ(deconv._stride[X_AXIS], 5u);
    deconv._stride_y = 6u;
    ASSERT_EQ(deconv._stride[Y_AXIS], 6u);
    deconv._dilation_x = 7u;
    ASSERT_EQ(deconv._dilation[X_AXIS], 7u);
    deconv._dilation_y = 8u;
    ASSERT_EQ(deconv._dilation[Y_AXIS], 8u);

    deconv._kernel[X_AXIS] = 8u;
    ASSERT_EQ(deconv._kernel_x, 8u);
    deconv._kernel[Y_AXIS] = 7u;
    ASSERT_EQ(deconv._kernel_y, 7u);
    deconv._padding[X_AXIS] = 6u;
    ASSERT_EQ(deconv._padding_x, 6u);
    deconv._padding[Y_AXIS] = 5u;
    ASSERT_EQ(deconv._padding_y, 5u);
    deconv._stride[X_AXIS] = 4u;
    ASSERT_EQ(deconv._stride_x, 4u);
    deconv._stride[Y_AXIS] = 3u;
    ASSERT_EQ(deconv._stride_y, 3u);
    deconv._dilation[X_AXIS] = 2u;
    ASSERT_EQ(deconv._dilation_x, 2u);
    deconv._dilation[Y_AXIS] = 9u;
    ASSERT_EQ(deconv._dilation_y, 9u);
}

TEST_F(LayersTests, poolIRv2BackwardCompatibility) {
    PoolingLayer pool(getDefaultParamsForLayer());
    ASSERT_NO_THROW(pool._kernel[X_AXIS]);
    ASSERT_NO_THROW(pool._kernel[Y_AXIS]);
    ASSERT_NO_THROW(pool._padding[X_AXIS]);
    ASSERT_NO_THROW(pool._padding[Y_AXIS]);
    ASSERT_NO_THROW(pool._stride[X_AXIS]);
    ASSERT_NO_THROW(pool._stride[Y_AXIS]);

    pool._kernel_x = 9u;
    ASSERT_EQ(pool._kernel[X_AXIS], 9u);
    pool._kernel_y = 2u;
    ASSERT_EQ(pool._kernel[Y_AXIS], 2u);
    pool._padding_x = 3u;
    ASSERT_EQ(pool._padding[X_AXIS], 3u);
    pool._padding_y = 4u;
    ASSERT_EQ(pool._padding[Y_AXIS], 4u);
    pool._stride_x = 5u;
    ASSERT_EQ(pool._stride[X_AXIS], 5u);
    pool._stride_y = 6u;
    ASSERT_EQ(pool._stride[Y_AXIS], 6u);

    pool._kernel[X_AXIS] = 8u;
    ASSERT_EQ(pool._kernel_x, 8u);
    pool._kernel[Y_AXIS] = 7u;
    ASSERT_EQ(pool._kernel_y, 7u);
    pool._padding[X_AXIS] = 6u;
    ASSERT_EQ(pool._padding_x, 6u);
    pool._padding[Y_AXIS] = 5u;
    ASSERT_EQ(pool._padding_y, 5u);
    pool._stride[X_AXIS] = 4u;
    ASSERT_EQ(pool._stride_x, 4u);
    pool._stride[Y_AXIS] = 3u;
    ASSERT_EQ(pool._stride_y, 3u);
}

TEST_F(LayersTests, canGetPadBeginForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    PropertyVector<unsigned> ref{{1, 2}};
    layer._padding = ref;

    auto allPads = getPaddings(layer);

    ASSERT_EQ(allPads.begin, ref);
}

TEST_F(LayersTests, canGetPadEndForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    PropertyVector<unsigned> ref{{1, 2}};
    layer._pads_end = ref;

    auto allPads = getPaddings(layer);

    ASSERT_EQ(allPads.end, ref);
}

TEST_F(LayersTests, canGetPad3DBeginForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    PropertyVector<unsigned> ref;
    ref.insert(X_AXIS, 1);
    ref.insert(Y_AXIS, 2);
    ref.insert(Z_AXIS, 3);
    layer._padding = ref;

    auto allPads = getPaddings(layer);

    ASSERT_EQ(allPads.begin, ref);
}

TEST_F(LayersTests, canGetPad3DEndForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    PropertyVector<unsigned> ref;
    ref.insert(X_AXIS, 1);
    ref.insert(Y_AXIS, 2);
    ref.insert(Z_AXIS, 3);
    layer._pads_end = ref;

    auto allPads = getPaddings(layer);

    ASSERT_EQ(allPads.end, ref);
}

TEST_F(LayersTests, returnDefaultPadForEmptyConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    auto allPads = getPaddings(layer);
    PropertyVector<unsigned> ref_begin(2, 0u);
    PropertyVector<unsigned> ref_end;
    ASSERT_EQ(allPads.begin, ref_begin);
    ASSERT_EQ(allPads.end, ref_end);
}

TEST_F(LayersTests, returnEmptyPadForValidPadConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "valid";
    auto allPads = getPaddings(layer);
    PropertyVector<unsigned> ref(2,0u);
    ASSERT_EQ(allPads.begin, ref);
    ASSERT_EQ(allPads.end, ref);

    PropertyVector<unsigned> ref3D(2,0u);
    layer._kernel.insert(Z_AXIS, 0u);
    ASSERT_EQ(allPads.begin, ref3D);
    ASSERT_EQ(allPads.end, ref3D);
}

TEST_F(LayersTests, throwOnSamePadForEmptyConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    ASSERT_THROW(getPaddings(layer), Exception);
}

TEST_F(LayersTests, throwOnInvalidDimsSamePadForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    auto emptyData = std::make_shared<InferenceEngine::Data>("", TensorDesc(Precision::UNSPECIFIED, Layout::ANY));
    layer.insData.push_back(emptyData);
    ASSERT_THROW(getPaddings(layer), Exception);
}

TEST_F(LayersTests, throwOn2DSamePadForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", TensorDesc(Precision::UNSPECIFIED, SizeVector{ 1, 1 }, Layout::NC));
    layer.insData.push_back(notEmptyData);
    ASSERT_THROW(getPaddings(layer), Exception);
}

TEST_F(LayersTests, throwWithNotEnoughParamsSamePadForConvolution) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", TensorDesc(Precision::UNSPECIFIED, SizeVector{ 4, 3, 2, 1 }, Layout::ANY));
    layer.insData.push_back(notEmptyData);
    ASSERT_NO_THROW(getPaddings(layer));

    auto notEmptyData3D = std::make_shared<InferenceEngine::Data>("", TensorDesc(Precision::UNSPECIFIED, SizeVector{ 5, 4, 3, 2, 1 }, Layout::NCDHW));
    layer._kernel.insert(Z_AXIS, 0u);
    layer.insData[0] = notEmptyData3D;
    ASSERT_NO_THROW(getPaddings(layer));
}

// parameters are from real model, like Mobilenet-SSD
TEST_F(LayersTests, canGetSamePadForConvolutionEvenInput) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 160, 160}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._dilation = PropertyVector<unsigned>{{1, 1}};
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, PropertyVector<unsigned>(2, 0));
    ASSERT_EQ(pad.end, PropertyVector<unsigned>(2, 1));
}

// parameters are from real model, like V-Net
TEST_F(LayersTests, canGetSamePadForConvolutionEvenInput3D) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 6, 190, 190, 20}, Layout::NCDHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._dilation.insert(X_AXIS, 1u);
    layer._dilation.insert(Y_AXIS, 1u);
    layer._dilation.insert(Z_AXIS, 1u);
    layer._kernel.insert(X_AXIS, 5u);
    layer._kernel.insert(Y_AXIS, 5u);
    layer._kernel.insert(Z_AXIS, 5u);
    layer._stride.insert(X_AXIS, 1u);
    layer._stride.insert(Y_AXIS, 1u);
    layer._stride.insert(Z_AXIS, 1u);

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, PropertyVector<unsigned>(3, 2u));
    ASSERT_EQ(pad.end, PropertyVector<unsigned>(3, 2u));
}

// parameters are from real model, like Mobilenet-SSD
TEST_F(LayersTests, canGetSamePadForConvolutionOddInput) {
    ConvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 75, 75}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._dilation = PropertyVector<unsigned>{{1, 1}};
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};
    PropertyVector<unsigned> ref(2, 1);

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, ref);
    ASSERT_EQ(pad.end, ref);
}

TEST_F(LayersTests, canGetSamePadForDeConvolutionEvenInput) {
    DeconvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 160, 160}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._dilation = PropertyVector<unsigned>{{1, 1}};
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, PropertyVector<unsigned>(2, 0));
    ASSERT_EQ(pad.end, PropertyVector<unsigned>(2, 1));
}

TEST_F(LayersTests, canGetSamePadForDeConvolutionOddInput) {
    DeconvolutionLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 75, 75}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._dilation = PropertyVector<unsigned>{{1, 1}};
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};
    PropertyVector<unsigned> ref(2, 1);

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, ref);
    ASSERT_EQ(pad.end, ref);
}

TEST_F(LayersTests, canGetSamePadForPoolingEvenInput) {
    PoolingLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 160, 160}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, PropertyVector<unsigned>(2, 0));
    ASSERT_EQ(pad.end, PropertyVector<unsigned>(2, 1));
}

TEST_F(LayersTests, canGetSamePadForPoolingOddInput) {
    PoolingLayer layer(getDefaultParamsForLayer());
    layer.params["auto_pad"] = "same_upper";
    TensorDesc tensorDesc(Precision::UNSPECIFIED, SizeVector{1, 144, 75, 75}, Layout::NCHW);
    auto notEmptyData = std::make_shared<InferenceEngine::Data>("", tensorDesc);
    layer.insData.push_back(notEmptyData);
    layer._kernel = PropertyVector<unsigned>{{3, 3}};
    layer._stride = PropertyVector<unsigned>{{2, 2}};
    PropertyVector<unsigned> ref(2, 1);

    auto pad = getPaddings(layer);

    ASSERT_EQ(pad.begin, ref);
    ASSERT_EQ(pad.end, ref);
}


TEST_F(LayersTests, cannotGetPadForUnsupportedLayer) {
    FullyConnectedLayer layer(getDefaultParamsForLayer());
    ASSERT_ANY_THROW(getPaddingsImpl(layer));
}
