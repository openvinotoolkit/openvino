// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

#include <cmath>

#include <legacy/ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>

#include <vpu/utils/hw_disabled.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>

namespace vpu {

static
bool canTryHW(const ie::PoolingLayer::PoolType poolType,
              const int inputWidth,
              const int inputHeight,
              const int outputWidth,
              const int outputHeight,
              const int kernelSizeX,
              const int kernelSizeY,
              const int kernelStrideX,
              const int kernelStrideY,
              const int padLeft,
              const int padRight,
              const int padTop,
              const int padBottom,
              const std::string& autoPad,
              const bool excludePad,
              const bool hwOptimization,
              const bool hwDisabled) {
    auto tryHW = hwOptimization;

    // HW restrictions
    if (kernelStrideX != kernelStrideY) {
        tryHW = false;
    }

    // check if HW pooling has correct output size
    {
        int iw = inputWidth;
        int ih = inputHeight;

        int ow = outputWidth;
        int oh = outputHeight;

        // take additional hw paddings into account
        if ((iw % 2 == 1) && (kernelSizeX % 2 == 0) && (padRight == 0)) iw++;
        if ((ih % 2 == 1) && (kernelSizeY % 2 == 0) && (padBottom == 0)) ih++;

        int tempX = iw + (padLeft + padRight) - kernelSizeX;
        int tempY = ih + (padBottom + padTop) - kernelSizeY;

        int outWidthWithOutCeil  = (tempX + kernelStrideX) / kernelStrideX;
        int outHeightWithOutCeil = (tempY + kernelStrideY) / kernelStrideY;

        int outWidthWithCeil  = static_cast<int>(std::ceil(static_cast<double>(tempX) / kernelStrideX + 1));
        int outHeightWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempY) / kernelStrideY + 1));

        if ((ow != outWidthWithCeil) && (ow != outWidthWithOutCeil)) {
            tryHW = false;
        }

        if ((oh != outHeightWithCeil) && (oh != outHeightWithOutCeil)) {
            tryHW = false;
        }
    }

    // HW restrictions
    if (kernelSizeX > 15 ||
        kernelSizeY > 15 ||
        kernelStrideX > 8) {
        tryHW = false;
    }

    // TODO: 3x3s2 Avg pooling is not supported by HW
    if (kernelSizeX == 3 && kernelSizeY == 3 && kernelStrideX == 2 && poolType == ie::PoolingLayer::AVG) {
        tryHW = false;
    }

    // TODO: Avg pooling with even kernel size and odd input is not supported
    if ((kernelSizeX % 2 == 0 || kernelSizeY % 2 == 0)) {
        if (inputWidth % 2 == 1 || inputHeight % 2 == 1) {
            if (poolType == ie::PoolingLayer::PoolType::AVG) {
                tryHW = false;
            }
        }
    }

    // TODO : 5x5s3 Avg pooling hangs device
    if (kernelSizeX == 5 && kernelSizeY == 5 && kernelStrideX == 3 && poolType == ie::PoolingLayer::PoolType::AVG) {
        tryHW = false;
    }

    // TODO : 2x2s2 1278x718 HW MAX pool works worse than SW version
    if ((kernelSizeX % 2 == 0 || kernelSizeY % 2 == 0)) {
        if (inputWidth > 1000 || inputHeight > 700) {
            tryHW = false;
        }
    }

    // TODO: 3x3s2 Max Pooling with [2,2] padding is not supported by HW
    if (kernelSizeX == 3 && kernelSizeY == 3 &&
        kernelStrideX == 2 && kernelStrideY == 2 &&
        poolType == ie::PoolingLayer::MAX &&
        padLeft == 0 && padTop == 0 &&
        padRight == 2 && padBottom == 2) {
        tryHW = false;
    }

    //  FIX #14949, enable HW AVG pooling, need SW postproc
    //  HW AVG pooling will output wrong results in borders when excludePad=true
    bool hasPad = padLeft || padTop || padRight || padBottom;
    if (excludePad && hasPad && poolType == ie::PoolingLayer::PoolType::AVG) {
        // Only apply to small output tensors for now
        // May need to loose the condition if accuracy issues are met
        if (outputWidth <= 5 && outputHeight <= 5) {
            tryHW = false;
        }
    }

    // FIX #16406, #18639 AVG pooling result is always 0 in case of 1x1 kernel
    if (kernelSizeX == 1 && kernelSizeY == 1 && poolType == ie::PoolingLayer::PoolType::AVG) {
        tryHW = false;
    }

    // FIX #18761 - WA disble HW pooling for below case
    // ToDo: evaluate exact problem, more cases and proper fix
    if (kernelSizeX == 2 && kernelSizeY == 2 &&
        kernelStrideX == 1 && kernelStrideY == 1 &&
        (outputWidth % 2) == 0 &&
        (outputHeight % 2) == 0 &&
        autoPad == "same_upper") {
        tryHW = false;
    }

    // #28057: custom sample hangs if stressed
    if (inputWidth == 382 && inputHeight == 214 &&
        kernelSizeX == 2 && kernelSizeY == 2 &&
        kernelStrideX == 2 && kernelStrideY ==2 &&
        poolType == ie::PoolingLayer::MAX) {
        tryHW = false;
    }

    if (hwDisabled) {
        tryHW = false;
    }

    return tryHW;
}

static
void parsePool2D(const     Model      & model,
                 const ie::CNNLayerPtr& layer,
                 const     Data       & input,
                 const     Data       & output) {
    //
    // Extract parameters
    //

    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    VPU_THROW_UNLESS(poolLayer != nullptr, "failed dynamic cast to PoolingLayer");

    int kernelSizeX = poolLayer->_kernel_x;
    int kernelSizeY = poolLayer->_kernel_y;

    int kernelStrideX = poolLayer->_stride_x;
    int kernelStrideY = poolLayer->_stride_y;

    auto paddings  = getPaddings(*poolLayer);
    int  padLeft   = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int  padRight  = paddings.end.exist(ie::X_AXIS)   ? paddings.end[ie::X_AXIS]   : padLeft;
    int  padTop    = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int  padBottom = paddings.end.exist(ie::Y_AXIS)   ? paddings.end[ie::Y_AXIS]   : padTop;

    // for old IR's IE doesn't return valid padding. Fix paddings
    {
        int iw = input->desc().dim(Dim::W);
        int ih = input->desc().dim(Dim::H);

        int ow = output->desc().dim(Dim::W);
        int oh = output->desc().dim(Dim::H);

        int expectedIW = (ow - 1)*kernelStrideX + kernelSizeX;
        int expectedOW = (oh - 1)*kernelStrideY + kernelSizeY;

        if (expectedIW > iw + padLeft + padRight) {
            padRight  = expectedIW - (iw + padLeft);
        }

        if (expectedOW > ih + padTop + padBottom) {
            padBottom = expectedOW - (ih + padTop);
        }
    }

    auto poolType = poolLayer->_type;

    auto excludePad = poolLayer->_exclude_pad;

    auto autoPad = poolLayer->_auto_pad;

    auto stageType = StageType::None;
    if (poolType == ie::PoolingLayer::MAX) {
        stageType = StageType::StubMaxPool;
    } else if (poolType == ie::PoolingLayer::AVG) {
        stageType = StageType::StubAvgPool;
    } else {
        VPU_THROW_EXCEPTION << "Pooling Layer " << poolLayer->name << " has unsupported type: " << poolType;
    }

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();
    bool hwOptimization = env.config.get<HwAccelerationOption>();
    bool hwDisabled = HwDisabled(env.config, layer->name);

    int inputWidth = input->desc().dim(Dim::W);
    int inputHeight = input->desc().dim(Dim::H);
    int outputWidth = output->desc().dim(Dim::W);
    int outputHeight = output->desc().dim(Dim::H);

    // kernelStrideY doesn't matter when kernelSizeY==InputSizeY, change it to try HW in 1D case
    if (kernelSizeY == inputHeight + padTop + padBottom)
        kernelStrideY = kernelStrideX;

    bool tryHW = canTryHW(poolType,
                          inputWidth,
                          inputHeight,
                          outputWidth,
                          outputHeight,
                          kernelSizeX,
                          kernelSizeY,
                          kernelStrideX,
                          kernelStrideY,
                          padLeft,
                          padRight,
                          padTop,
                          padBottom,
                          autoPad,
                          excludePad,
                          hwOptimization,
                          hwDisabled);

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        layer->name,
        stageType,
        layer,
        {input},
        {output});

    stage->attrs().set<int>("kernelSizeX", kernelSizeX);
    stage->attrs().set<int>("kernelSizeY", kernelSizeY);

    stage->attrs().set<int>("kernelStrideX", kernelStrideX);
    stage->attrs().set<int>("kernelStrideY", kernelStrideY);

    stage->attrs().set<int>("padLeft", padLeft);
    stage->attrs().set<int>("padRight", padRight);
    stage->attrs().set<int>("padTop", padTop);
    stage->attrs().set<int>("padBottom", padBottom);

    stage->attrs().set<bool>("excludePad", excludePad);

    stage->attrs().set<bool>("tryHW", tryHW);
}

//----------------------------------------------------------------------

enum PoolNDMethod   { PoolND_max = 1, PoolND_avg = 2 };

enum PoolNDRounding { PoolND_floor = 3, PoolND_ceil  = 4 };

namespace {

class PoolNDStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PoolNDStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto outputOrder = input(0)->desc().dimsOrder();
        int nDims = outputOrder.numDims();
        if (nDims == 3 || nDims == 4) {
            outputOrder.moveDim(Dim::C, 2);  // ->NCHW, or ->CHW
        } else if (nDims == 5) {
            outputOrder.moveDim(Dim::C, 3);  // ->NCDHW
        } else {
            VPU_THROW_UNLESS(3 <= nDims && nDims <= 5, "unsupported nDims=%d", nDims);
        }
        orderInfo.setOutput(outputEdge(0), outputOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::NeedMax;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto inputValues = input(0);
        auto outputValues = output(0);

        inputValues->serializeBuffer(serializer);
        outputValues->serializeBuffer(serializer);
    }

    using PV = InferenceEngine::PropertyVector<unsigned int>;

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto kernel_shape = attrs().get<PV>("kernel_shape");
        auto pads_begin   = attrs().get<PV>("pads_begin");
        auto pads_end     = attrs().get<PV>("pads_end");
        auto strides      = attrs().get<PV>("strides");

        auto interleaved    = attrs().get<int>("interleaved");
        auto pooling_method = attrs().get<int>("pooling_method");
        auto rounding_type  = attrs().get<int>("rounding_type");
        auto exclude_pad    = attrs().get<int>("exclude_pad");

        append_pv(serializer, kernel_shape);
        append_pv(serializer, pads_begin);
        append_pv(serializer, pads_end);
        append_pv(serializer, strides);

        append_i(serializer, interleaved);
        append_i(serializer, pooling_method);
        append_i(serializer, rounding_type);
        append_i(serializer, exclude_pad);
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
void parsePoolND(const     Model      & model,
                 const ie::CNNLayerPtr& layer,
                 const     Data       & input,
                 const     Data       & output) {
    //
    // Check layer parameters
    //

    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    VPU_THROW_UNLESS(poolLayer != nullptr, "failed dynamic cast to PoolingLayer");

    auto kernel_shape = poolLayer->_kernel;
    int kernel_ndims = static_cast<int>(kernel_shape.size());
    // Yet, only 3D kernel supported (NCDHW)
    // Later, if support 4D, 5D, etc, please
    // check if (kernelNDims >= 3), so that
    // 2D case is supported separately with
    // parsePool2D() function
    VPU_THROW_UNLESS(kernel_ndims == 3, "unsupported kernel ndims=%d", kernel_ndims);

    auto paddings = getPaddings(*poolLayer);
    auto pads_begin = paddings.begin;
    auto pads_end   = paddings.end;
    VPU_THROW_UNLESS(pads_begin.size() == kernel_ndims,
                     "incompatible pad ndims: actual=%lu, expected=%d",
                     pads_begin.size(), kernel_ndims);
    VPU_THROW_UNLESS(pads_end.size() == kernel_ndims,
                     "incompatible pad ndims: actual=%lu, expected=%d",
                     pads_end.size(), kernel_ndims);

    auto strides = poolLayer->_stride;
    VPU_THROW_UNLESS(strides.size() == kernel_ndims,
                     "incompatible stride ndims: actual=%lu, expected=%d",
                     strides.size(), kernel_ndims);

    int input_ndims = input->desc().numDims();
    int output_ndims = output->desc().numDims();

    VPU_THROW_UNLESS(input_ndims == output_ndims,
                     "incompatible input and output ndims: input ndims=%d, output ndims=%d",
                     input_ndims, output_ndims);
    VPU_THROW_UNLESS(input_ndims == kernel_ndims + 2,
                     "input must have batch and channels, but: input ndims=%d, kernel ndims=%d",
                     input_ndims, kernel_ndims);

    VPU_THROW_UNLESS(input->desc().type() == DataType::FP16, "unsupported input data type");
    VPU_THROW_UNLESS(output->desc().type() == DataType::FP16, "unsupported output data type");

    int input_channels = input->desc().dim(Dim::C);
    int output_channels = output->desc().dim(Dim::C);
    VPU_THROW_UNLESS(input_channels == output_channels,
                     "numbers of channels must be equal: input channels=%d, output channels=%d",
                     input_channels, output_channels);

    int input_batch = input->desc().dim(Dim::N);
    int output_batch = output->desc().dim(Dim::N);
    VPU_THROW_UNLESS(input_batch == output_batch,
                     "incompatible batch sizes: input batch=%d, output batch=%d",
                     input_batch, output_batch);

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
    for (int i = 0; i < kernel_ndims; i++) {
        int expected_output_shape_i = (input_shape[i]
                                       + pads_begin[i] + pads_end[i]
                                       - kernel_shape[i])
                                    / strides[i] + 1;
        VPU_THROW_UNLESS(output_shape[i] == expected_output_shape_i,
                         "failed check of output shape: i=%d, actual=%d, expected=%d",
                         i, output_shape[i] == expected_output_shape_i);
    }

    int interleaved = 0;

    int pooling_method;
    if (poolLayer->_type == ie::PoolingLayer::MAX) {
        pooling_method = PoolND_max;
    } else if (poolLayer->_type == ie::PoolingLayer::AVG) {
        pooling_method = PoolND_avg;
    } else {
        VPU_THROW_EXCEPTION << "Pooling Layer " << poolLayer->name
                   << " has unsupported type: " << poolLayer->_type;
    }

    // TODO: Check rounding type! (after this becomes possible)
    // Yet, ie::PoolingLayer doesn't supporting such attribute,
    // so by default we assume rounding type is `floor`
    int rounding_type = PoolND_floor;

    int exclude_pad = poolLayer->_exclude_pad ? 1 : 0;

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();
    bool hwOptimization = env.config.get<HwAccelerationOption>();
    bool hwDisabled = HwDisabled(env.config, layer->name);

    bool tryHW = canTryHW(poolLayer->_type,
                          input_shape[0],
                          input_shape[1],
                          output_shape[0],
                          output_shape[1],
                          kernel_shape[0],
                          kernel_shape[1],
                          strides[0],
                          strides[1],
                          pads_begin[0],
                          pads_end[0],
                          pads_begin[1],
                          pads_end[1],
                          poolLayer->_auto_pad,
                          poolLayer->_exclude_pad,
                          hwOptimization,
                          hwDisabled);
    int try_hw = tryHW ? 1 : 0;

    //
    // Add new stage
    //

    auto stage = model->addNewStage<PoolNDStage>(
        layer->name,
        StageType::PoolND,
        layer,
        {input},
        {output});

    stage->attrs().set("kernel_shape", kernel_shape);
    stage->attrs().set("pads_begin",   pads_begin);
    stage->attrs().set("pads_end",     pads_end);
    stage->attrs().set("strides",      strides);

    stage->attrs().set("interleaved",    interleaved);
    stage->attrs().set("pooling_method", pooling_method);
    stage->attrs().set("rounding_type",  rounding_type);
    stage->attrs().set("exclude_pad",    exclude_pad);

    stage->attrs().set("try_hw", try_hw);
}

//----------------------------------------------------------------------

void FrontEnd::parsePooling(const     Model      & model,
                            const ie::CNNLayerPtr& layer,
                            const     DataVector & inputs,
                            const     DataVector & outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1, "number of inputs must be equal to 1, but it equals to %lu", inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1, "number of outputs must be equal to 1, but it equals to %lu", outputs.size());

    auto input = inputs[0];
    auto output = outputs[0];

    if (input->desc().numDims() < 3 || input->desc().numDims() > 5) {
        VPU_THROW_FORMAT("Pooling supports only 3D or 4D or 5D input, but input number of dims=%d",
                         input->desc().numDims());
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_FORMAT("Pooling supports only same num dims in input and output"
                         ", but input ndims=%d and output ndims=%d",
                         input->desc().numDims(), output->desc().numDims());
    }

    bool is2D = input->desc().numDims() == 3 ||
                input->desc().numDims() == 4;  // CHW or NCHW, but not NCDWH or 6D or ...

    if (is2D) {
        parsePool2D(model, layer, input, output);
    } else {
        parsePoolND(model, layer, input, output);
    }
}

//----------------------------------------------------------------------

Stage StageBuilder::addPoolingStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const ie::PoolingLayer::PoolType& poolType) {
    //
    // Check parameters: only 2D pooling is supported (yet)
    //
    VPU_THROW_UNLESS(input->desc().dimsOrder() == DimsOrder::NCHW, "unsupported input dims order");
    VPU_THROW_UNLESS(output->desc().dimsOrder() == DimsOrder::NCHW, "unsupported output dims order");

    //
    // Check pooling method: only "avg" or "max" pooling (yet)
    //
    StageType stageType;
    if (poolType == ie::PoolingLayer::PoolType::AVG) {
        stageType = StageType::StubAvgPool;
    } else if (poolType == ie::PoolingLayer::PoolType::MAX) {
        stageType = StageType::StubMaxPool;
    } else {
        stageType = StageType::Empty;
        VPU_THROW_UNLESS(poolType == ie::PoolingLayer::PoolType::AVG ||
                         poolType == ie::PoolingLayer::PoolType::MAX,
                         "unsupported pooling type: %d", poolType);
    }

    //
    // Add 2D pooling stage (stub)
    //
    auto stage = model->addNewStage<StubStage>(name,
                                               stageType,
                                               layer,
                                               {input},
                                               {output});
    return stage;
}

}  // namespace vpu
