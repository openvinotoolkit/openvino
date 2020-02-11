// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <tuple>
#include <set>

#include <ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>

namespace vpu {

using InferenceEngine::CNNLayerPtr;

static
void parseConv2D(const Model      & model,
                 const CNNLayerPtr& layer,
                 const Data       & input,
                 const Data       & output,
                       Data       & weights,
                       Data       & biases);

static
void parseConvND(const Model      & model,
                 const CNNLayerPtr& layer,
                 const Data       & input,
                 const Data       & output,
                 const Data       & weights,
                 const Data       & biases);

void FrontEnd::parseConvolution(const Model& model, const CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    if (input->desc().numDims() < 3 || input->desc().numDims() > 5) {
        VPU_THROW_EXCEPTION << "Convolution supports only 3D or 4D or 5D input";
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_EXCEPTION << "Convolution supports only same num dims in input and output";
    }

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    bool is2D = input->desc().numDims() == 3 ||
                input->desc().numDims() == 4;  // CHW or NCHW, not NCDWH or 6D or ...
    if (is2D) {
        parseConv2D(model, layer, input, output, weights, biases);
    } else {
        parseConvND(model, layer, input, output, weights, biases);
    }
}

//----------------------------------------------------------------------

static
void parseConv2D(const Model      & model,
                 const CNNLayerPtr& layer,
                 const Data       & input,
                 const Data       & output,
                       Data       & weights,
                       Data       & biases) {
    //
    // Extract parameters
    //

    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    int kernelSizeX = convLayer->_kernel_x;
    int kernelSizeY = convLayer->_kernel_y;

    int kernelStrideX = convLayer->_stride_x;
    int kernelStrideY = convLayer->_stride_y;

    auto paddings = getPaddings(*convLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = convLayer->_dilation_x;
    int dilationY = convLayer->_dilation_y;

    int groupSize = convLayer->_group;

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();

    auto tryHW = env.config.hwOptimization;

    if (kernelStrideX != kernelStrideY) {
        tryHW = false;
    }

    // TODO: support dilated convolution
    if ((dilationX != 1 || dilationY != 1) && (!env.config.hwDilation)) {
        tryHW = false;
    }

    if (kernelSizeX > 15 || kernelSizeY > 15 || kernelStrideX > 8) {
        tryHW = false;
    }

    if (env.config.hwDisabled(layer->name)) {
        tryHW = false;
    }

    if (output->desc().numDims() < 4) {
        tryHW = false;
    }

    //
    // Create const datas
    //

    IE_ASSERT(weights->desc().totalDimSize() >=
              kernelSizeX * kernelSizeY * (input->desc().dim(Dim::C) / groupSize) * output->desc().dim(Dim::C));

    auto weightsDesc =
        DataDesc({
            kernelSizeX,
            kernelSizeY,
            input->desc().dim(Dim::C) / groupSize,
            output->desc().dim(Dim::C)
        });

    weights = model->duplicateData(
        weights,
        "@conv",
        weightsDesc);

    if (biases->usage() != DataUsage::Fake) {
        IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
        biases = model->duplicateData(
            biases,
            "@conv",
            DataDesc({output->desc().dim(Dim::C)}));
    }

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        layer->name,
        StageType::StubConv,
        layer,
        {input, weights, biases, model->addFakeData()},
        {output});

    stage->attrs().set<int>("kernelSizeX", kernelSizeX);
    stage->attrs().set<int>("kernelSizeY", kernelSizeY);

    stage->attrs().set<int>("kernelStrideX", kernelStrideX);
    stage->attrs().set<int>("kernelStrideY", kernelStrideY);

    stage->attrs().set<int>("padLeft", padLeft);
    stage->attrs().set<int>("padRight", padRight);
    stage->attrs().set<int>("padTop", padTop);
    stage->attrs().set<int>("padBottom", padBottom);

    stage->attrs().set<int>("dilationX", dilationX);
    stage->attrs().set<int>("dilationY", dilationY);

    stage->attrs().set<int>("groupSize", groupSize);

    stage->attrs().set<bool>("tryHW", tryHW);
}

//----------------------------------------------------------------------

namespace {

class ConvNDStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConvNDStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto outputOrder = input(0)->desc().dimsOrder();
        int nDims = outputOrder.numDims();
        if (nDims == 3 || nDims == 4) {
            outputOrder.moveDim(Dim::C, 2);  // ->NCHW, or ->CHW
        } else if (nDims == 5) {
            outputOrder.moveDim(Dim::C, 3);  // ->NCDHW
        } else {
            IE_ASSERT(3 <= nDims && nDims <= 5);
        }
        orderInfo.setOutput(outputEdge(0), outputOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        if (type() != StageType::DepthConv) {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NeedMax;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
            {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
            {{DataType::FP16}});
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto inputValues = input(0);
        auto inputWeights = input(1);
        auto inputBiases = input(2);
        auto outputValues = output(0);

        inputValues->serializeNewBuffer(serializer);
        outputValues->serializeNewBuffer(serializer);
        inputWeights->serializeNewBuffer(serializer);
        inputBiases->serializeNewBuffer(serializer);
    }

    using PV = InferenceEngine::PropertyVector<unsigned int>;

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto pads_begin = attrs().get<PV>("pads_begin");
        auto pads_end   = attrs().get<PV>("pads_end");

        auto strides    = attrs().get<PV>("strides");
        auto dilations  = attrs().get<PV>("dilations");

        auto groups     = attrs().get<int>("groups");

        append_pv(serializer, pads_begin);
        append_pv(serializer, pads_end);

        append_pv(serializer, strides);
        append_pv(serializer, dilations);

        append_i(serializer, groups);
    }

    static void append_pv(BlobSerializer& serializer, const PV& pv) {
        int ndims = pv.size();
        append_i(serializer, ndims);
        for (int i = 0; i < ndims; i++) {
            append_i(serializer, pv[i]);
        }
    }

    static void append_i(BlobSerializer& serializer, int i) {
        serializer.append(static_cast<int32_t>(i));
    }
};

}  // namespace

static
void parseConvND(const Model      & model,
                 const CNNLayerPtr& layer,
                 const Data       & input,
                 const Data       & output,
                 const Data       & weights,
                 const Data       & biases) {
    //
    // Check layer parameters
    //

    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    auto kernelShape = convLayer->_kernel;
    int kernelNDims = kernelShape.size();
    // Yet, only 3D kernel supported (NCDHW)
    // Later, if support 4D, 5D, etc, please
    // check if (kernelNDims >= 3), so that
    // 2D case is supported separately with
    // parseConv2D() function
    IE_ASSERT(kernelNDims == 3);

    auto paddings = getPaddings(*convLayer);
    auto pads_begin = paddings.begin;
    auto pads_end   = paddings.end;
    IE_ASSERT(pads_begin.size() == pads_end.size());
    IE_ASSERT(pads_begin.size() == kernelShape.size());

    auto strides = convLayer->_stride;
    IE_ASSERT(strides.size() == kernelShape.size());

    auto dilations = convLayer->_dilation;
    IE_ASSERT(dilations.size() == kernelShape.size());

    int output_channels = convLayer->_out_depth;
    IE_ASSERT(output_channels > 0);

    int groups = convLayer->_group;
    IE_ASSERT(groups > 0);

    int inputNDims = input->desc().numDims();
    int outputNDims = output->desc().numDims();
    int weightsNDims = weights->desc().numDims();
    int biasesNDims = biases->desc().numDims();

    IE_ASSERT(inputNDims == outputNDims);
    IE_ASSERT(inputNDims == kernelNDims + 2);  // NCDHW, 6D, ...
    IE_ASSERT(weightsNDims == 1);  // need to reshape, see below
    IE_ASSERT(biasesNDims == 1);

    int input_channels = input->desc().dim(Dim::C);
    IE_ASSERT(output_channels == output->desc().dim(Dim::C));
    IE_ASSERT(input_channels % groups == 0);
    IE_ASSERT(output_channels % groups == 0);
    IE_ASSERT(output_channels / groups == biases->desc().dim(Dim::C));

    // Checking spacial dimensions of output...
    // NB: Note, that input/output shape arrays
    // have inverse order, like {W, H, D, C, N}
    int  input_width  =  input->desc().dim(Dim::W);
    int output_width  = output->desc().dim(Dim::W);
    int  input_height =  input->desc().dim(Dim::H);
    int output_height = output->desc().dim(Dim::H);
    int  input_depth  =  input->desc().dim(Dim::D);
    int output_depth  = output->desc().dim(Dim::D);
    int  input_shape[] = { input_width,  input_height,  input_depth};
    int output_shape[] = {output_width, output_height, output_depth};
    for (int i = 0; i < kernelNDims; i++) {
        int dilated_kernel_shape_i = dilations[i] * (kernelShape[i] - 1) + 1;
        int expected_output_shape_i = (input_shape[i]
                                       + pads_begin[i] + pads_end[i]
                                       - dilated_kernel_shape_i)
                                    / strides[i] + 1;
        IE_ASSERT(output_shape[i] == expected_output_shape_i);
    }

    IE_ASSERT(input->desc().type() == DataType::FP16);
    IE_ASSERT(output->desc().type() == DataType::FP16);
    IE_ASSERT(weights->desc().type() == DataType::FP16);
    IE_ASSERT(biases->desc().type() == DataType::FP16);

    //
    // Reshape weights, check biases
    //

    int kernelTotalElems = 1;
    for (int i = 0; i < kernelNDims; i++) {
        kernelTotalElems *= kernelShape[i];
    }

    int weightsTotalElems = kernelTotalElems *
                            (input_channels / groups) *
                            (output_channels / groups);
    IE_ASSERT(weights->desc().totalDimSize() == weightsTotalElems);

    std::vector<int> weightsShape(kernelNDims + 2);
    for (int i = 0; i < kernelNDims; i++) {
        weightsShape[i] = kernelShape[i];
    }
    weightsShape[kernelNDims + 0] =  input_channels / groups;
    weightsShape[kernelNDims + 1] = output_channels / groups;

    DataDesc weightsDesc(weightsShape);
    auto weightsReshaped = model->duplicateData(weights, "@conv3d", weightsDesc);

    IE_ASSERT(biases->desc().totalDimSize() == output_channels / groups);

    //
    // Add new stage
    //

    auto stage = model->addNewStage<ConvNDStage>(
        layer->name,
        StageType::ConvND,
        layer,
        {input, weightsReshaped, biases},
        {output});

    stage->attrs().set("pads_begin", pads_begin);
    stage->attrs().set("pads_end",   pads_end);

    stage->attrs().set("strides",    strides);
    stage->attrs().set("dilations",  dilations);

    stage->attrs().set("groups",     groups);
}

}  // namespace vpu
