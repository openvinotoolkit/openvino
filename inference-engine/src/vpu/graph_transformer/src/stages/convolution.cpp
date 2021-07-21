// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <tuple>
#include <set>

#include <legacy/ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/utils/hw_disabled.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>
#include <vpu/configuration/options/hw_dilation.hpp>

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

void FrontEnd::parseConvolution(const Model      & model,
                                const CNNLayerPtr& layer,
                                const DataVector & inputs,
                                const DataVector & outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1, "invalid number of inputs: %lu", inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1, "invalid number of outputs: %lu", outputs.size());

    auto input = inputs[0];
    auto output = outputs[0];

    if (input->desc().numDims() < 3 || input->desc().numDims() > 5) {
        VPU_THROW_FORMAT("Convolution supports only 3D or 4D or 5D input, but input number of dims=%d",
                         input->desc().numDims());
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_FORMAT("Convolution supports only same num dims in input and output"
                         ", but input ndims=%d and output ndims=%d",
                         input->desc().numDims(), output->desc().numDims());
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
bool canTryHW(const int outputNumDims,
              const int kernelSizeX,
              const int kernelSizeY,
              const int kernelStrideX,
              const int kernelStrideY,
              const int dilationX,
              const int dilationY,
              const bool hwOptimization,
              const bool hwDilation,
              const bool hwDisabled) {
    bool tryHW = hwOptimization;

    if (kernelStrideX != kernelStrideY) {
        tryHW = false;
    }

// TODO: support dilated convolution
    if ((dilationX != 1 || dilationY != 1) && (!hwDilation)) {
        tryHW = false;
    }

    // 1x16 convolution is split into two 1x8 convolutions in splitLargeKernelConv pass
    const bool KernelSizeCantBeSplit = !(kernelSizeX == 16 && kernelSizeY == 1);
    const bool KernelSizeTooLarge = (kernelSizeX > 15 || kernelSizeY > 15);
    if (KernelSizeTooLarge && KernelSizeCantBeSplit) {
        tryHW = false;
    }

    if (kernelStrideX > 8) {
        tryHW = false;
    }

    if (hwDisabled) {
        tryHW = false;
    }

    if (outputNumDims < 4) {
        tryHW = false;
    }

    return tryHW;
}

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
    VPU_THROW_UNLESS(convLayer != nullptr, "failed dynamic cast to ConvolutionLayer");

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

    // kernelStrideY doesn't matter when kernelSizeY==InputSizeY, change it to try HW in 1D case
    if (kernelSizeY == input->desc().dim(Dim::H) + padTop + padBottom)
        kernelStrideY = kernelStrideX;

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();

    bool tryHW = canTryHW(output->desc().numDims(),
                          kernelSizeX,
                          kernelSizeY,
                          kernelStrideX,
                          kernelStrideY,
                          dilationX,
                          dilationY,
                          env.config.get<HwAccelerationOption>(),
                          env.config.get<HwDilationOption>(),
                          HwDisabled(env.config, layer->name));

    //
    // Create const datas
    //

    int weightsActualSize = weights->desc().totalDimSize();
    int weightsExpectedSize = kernelSizeX * kernelSizeY *
                              (input->desc().dim(Dim::C) / groupSize) *
                              output->desc().dim(Dim::C);
    VPU_THROW_UNLESS(weightsActualSize >= weightsExpectedSize,
                     "too few actual weights: actual size=%d, expected size=%d",
                     weightsActualSize, weightsExpectedSize);

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
        int biasesActualSize = biases->desc().totalDimSize();
        int biasesExpectedSize = output->desc().dim(Dim::C);
        VPU_THROW_UNLESS(biasesActualSize >= biasesExpectedSize,
            "too few biases: actual size=%d, expected size=%d",
            biasesActualSize, biasesExpectedSize);
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
            VPU_THROW_UNLESS(3 <= nDims && nDims <= 5, "unsupported number of dims: %d", nDims);
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

        inputValues->serializeBuffer(serializer);
        outputValues->serializeBuffer(serializer);
        inputWeights->serializeBuffer(serializer);
        inputBiases->serializeBuffer(serializer);
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
        int ndims = static_cast<int>(pv.size());
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
    VPU_THROW_UNLESS(convLayer != nullptr, "failed dynamic cast to ConvolutionLayer");

    auto kernelShape = convLayer->_kernel;
    int kernelNDims = static_cast<int>(kernelShape.size());
    // Yet, only 3D kernel supported (NCDHW)
    // Later, if support 4D, 5D, etc, please
    // check if (kernelNDims >= 3), so that
    // 2D case is supported separately with
    // parseConv2D() function
    VPU_THROW_UNLESS(kernelNDims == 3, "unsupported number of kernel dims: %d", kernelNDims);

    auto paddings = getPaddings(*convLayer);
    auto pads_begin = paddings.begin;
    auto pads_end   = paddings.end;
    VPU_THROW_UNLESS(pads_begin.size() == pads_end.size(),
                     "number of dims must be equal: pads_begin ndims=%lu, pads_end ndims=%lu",
                     pads_begin.size(), pads_end.size());
    VPU_THROW_UNLESS(pads_begin.size() == kernelShape.size(),
                     "number of dims must equal: pads ndims=%lu, kernel ndims=%lu",
                     pads_begin.size(), kernelShape.size());

    auto strides = convLayer->_stride;
    VPU_THROW_UNLESS(strides.size() == kernelShape.size(),
                     "number of dims must equal: strides ndims=%lu, kernel ndims=%d",
                     strides.size(), kernelShape.size());

    auto dilations = convLayer->_dilation;
    VPU_THROW_UNLESS(dilations.size() == kernelShape.size(),
                     "number of dims must equal: dilations ndims=%lu, kernel ndims=%lu",
                     dilations.size(), kernelShape.size());

    int output_channels = convLayer->_out_depth;
    VPU_THROW_UNLESS(output_channels > 0, "invalid number of output channels: %d", output_channels);

    int groups = convLayer->_group;
    VPU_THROW_UNLESS(groups == 1, "number of groups=%d, but grouped 3D convolution is not supported", groups);

    int inputNDims = input->desc().numDims();
    int outputNDims = output->desc().numDims();
    int biasesNDims = biases->desc().numDims();

    VPU_THROW_UNLESS(inputNDims == outputNDims,
                     "number of dims must equal: input ndims=%d, output ndims=%d",
                     inputNDims, outputNDims);
    VPU_THROW_UNLESS(inputNDims == kernelNDims + 2,
                     "input must have 2 additional dims (for batch and channels), but: input ndims=%d, kernel ndims=%d",
                     inputNDims, kernelNDims);
    VPU_THROW_UNLESS(biasesNDims == 1, "biases must come as 1D array, but: biases ndims=%d", biasesNDims);

    int input_channels = input->desc().dim(Dim::C);
    VPU_THROW_UNLESS(output_channels == output->desc().dim(Dim::C),
                     "number of output channels must equal, but: expected=%d, actual=%d",
                     output_channels, output->desc().dim(Dim::C));
    VPU_THROW_UNLESS(input_channels % groups == 0,
                     "number of groups must divide the number of input channels, but: channels=%d, groups=%d",
                     input_channels, groups);
    VPU_THROW_UNLESS(output_channels % groups == 0,
                     "number of groups must divide the number of output channels, but: channels=%d, groups=%d",
                     output_channels, groups);
    VPU_THROW_UNLESS(output_channels / groups == biases->desc().dim(Dim::C),
                     "number of biases must equal to number of output channels per group, but: "
                     "channels per group=%d, biases=%d",
                     output_channels / groups, biases->desc().dim(Dim::C));

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
        VPU_THROW_UNLESS(output_shape[i] == expected_output_shape_i,
                         "output shape check failed: output_shape[%d]=%d, expected output_shape[%d]=%d",
                         i, output_shape[i], i, expected_output_shape_i);
    }

    VPU_THROW_UNLESS(input->desc().type() == DataType::FP16, "unsupported data type: %d", input->desc().type());
    VPU_THROW_UNLESS(output->desc().type() == DataType::FP16, "unsupported data type: %d", output->desc().type());
    VPU_THROW_UNLESS(weights->desc().type() == DataType::FP16, "unsupported data type: %d", weights->desc().type());
    VPU_THROW_UNLESS(biases->desc().type() == DataType::FP16, "unsupported data type: %d", biases->desc().type());

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
    VPU_THROW_UNLESS(weights->desc().totalDimSize() == weightsTotalElems,
                     "failed check of weights size: actual=%d, expected=%d",
                     weights->desc().totalDimSize(), weightsTotalElems);

    std::vector<int> weightsShape(kernelNDims + 2);
    for (int i = 0; i < kernelNDims; i++) {
        weightsShape[i] = kernelShape[i];
    }
    weightsShape[kernelNDims + 0] =  input_channels / groups;
    weightsShape[kernelNDims + 1] = output_channels / groups;

    DataDesc weightsDesc(weightsShape);
    auto weightsReshaped = model->duplicateData(weights, "@conv3d", weightsDesc);

    VPU_THROW_UNLESS(biases->desc().totalDimSize() == output_channels / groups,
                     "failed check of biases size: actual=%d, expected=%d",
                     biases->desc().totalDimSize(), output_channels / groups);

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();

    bool tryHW = canTryHW(outputNDims - 1,
                          kernelShape[0],
                          kernelShape[1],
                          strides[0],
                          strides[1],
                          dilations[0],
                          dilations[1],
                          env.config.get<HwAccelerationOption>(),
                          env.config.get<HwDilationOption>(),
                          HwDisabled(env.config, layer->name));

    int try_hw = tryHW ? 1 : 0;

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

    stage->attrs().set("try_hw",     try_hw);
}

//----------------------------------------------------------------------

Stage StageBuilder::addConvolutionStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const Data& weights,
        const Data& biases,
        const Data& scales) {
    //
    // Check parameters: only 2D convolution supported (yet)
    //
    VPU_THROW_UNLESS(input->desc().dimsOrder() == DimsOrder::NCHW, "unsupported dims order");
    VPU_THROW_UNLESS(output->desc().dimsOrder() == DimsOrder::NCHW, "unsupported dims order");

    //
    // Add 2D convolution stage (stub)
    //
    auto stage = model->addNewStage<StubStage>(
        name,
        StageType::StubConv,
        layer,
        {input, weights, biases, scales},
        {output});
    return stage;
}

}  // namespace vpu
