// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

#include <ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>

namespace vpu {

static
void parsePool2D(const     Model      & model,
                 const ie::CNNLayerPtr& layer,
                 const     Data       & input,
                 const     Data       & output) {
    //
    // Extract parameters
    //

    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

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

    //
    // Check if HW is applicable
    //

    const auto& env = CompileEnv::get();

    auto stageType = StageType::None;
    auto tryHW = env.config.hwOptimization;

    if (poolType == ie::PoolingLayer::MAX) {
        stageType = StageType::StubMaxPool;
    } else if (poolType == ie::PoolingLayer::AVG) {
        stageType = StageType::StubAvgPool;
    } else {
        VPU_THROW_EXCEPTION << "Pooling Layer " << poolLayer->name << " has unsupported type: " << poolType;
    }

    // HW restrictions
    if (kernelStrideX != kernelStrideY) {
        tryHW = false;
    }

    // check if HW pooling has correct output size
    {
        int iw = input->desc().dim(Dim::W);
        int ih = input->desc().dim(Dim::H);

        int ow = output->desc().dim(Dim::W);
        int oh = output->desc().dim(Dim::H);

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
        if (input->desc().dim(Dim::W) % 2 == 1 || input->desc().dim(Dim::H) % 2 == 1) {
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
        if (input->desc().dim(Dim::W) > 1000 || input->desc().dim(Dim::H) > 700) {
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
    if (excludePad && poolType == ie::PoolingLayer::PoolType::AVG) {
        if (output->desc().dim(Dim::W) == 5 &&
            output->desc().dim(Dim::H) == 5 &&
            kernelSizeX == 5 &&
            kernelSizeY == 5) {
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
        (output->desc().dim(Dim::W) & 1) == 0 &&
        (output->desc().dim(Dim::H) & 1) == 0 &&
        poolLayer->_auto_pad == "same_upper") {
        tryHW = false;
    }
    if (env.config.hwDisabled(layer->name)) {
        tryHW = false;
    }

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
            IE_ASSERT(3 <= nDims && nDims <= 5);
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

        inputValues->serializeNewBuffer(serializer);
        outputValues->serializeNewBuffer(serializer);
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
void parsePoolND(const     Model      & model,
                 const ie::CNNLayerPtr& layer,
                 const     Data       & input,
                 const     Data       & output) {
    //
    // Check layer parameters
    //

    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    auto kernel_shape = poolLayer->_kernel;
    int kernel_ndims = kernel_shape.size();
    // Yet, only 3D kernel supported (NCDHW)
    // Later, if support 4D, 5D, etc, please
    // check if (kernelNDims >= 3), so that
    // 2D case is supported separately with
    // parsePool2D() function
    IE_ASSERT(kernel_ndims == 3);

    auto paddings = getPaddings(*poolLayer);
    auto pads_begin = paddings.begin;
    auto pads_end   = paddings.end;
    IE_ASSERT(pads_begin.size() == kernel_ndims);
    IE_ASSERT(pads_end.size() == kernel_ndims);

    auto strides = poolLayer->_stride;
    IE_ASSERT(strides.size() == kernel_ndims);

    int input_ndims = input->desc().numDims();
    int output_ndims = output->desc().numDims();

    IE_ASSERT(input_ndims == output_ndims);
    IE_ASSERT(input_ndims == kernel_ndims + 2);  // NCDHW, 6D, ...

    IE_ASSERT(input->desc().type() == DataType::FP16);
    IE_ASSERT(output->desc().type() == DataType::FP16);

    int input_channels = input->desc().dim(Dim::C);
    int output_channels = output->desc().dim(Dim::C);
    IE_ASSERT(input_channels == output_channels);

    int input_batch = input->desc().dim(Dim::N);
    int output_batch = output->desc().dim(Dim::N);
    IE_ASSERT(input_batch == output_batch);

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
        IE_ASSERT(output_shape[i] == expected_output_shape_i);
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
}

//----------------------------------------------------------------------

void FrontEnd::parsePooling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    if (input->desc().numDims() < 3 || input->desc().numDims() > 5) {
        VPU_THROW_EXCEPTION << "Pooling supports only 3D or 4D or 5D input";
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_EXCEPTION << "Pooling supports only same num dims in input and output";
    }

    bool is2D = input->desc().numDims() == 3 ||
                input->desc().numDims() == 4;  // CHW or NCHW, but not NCDWH or 6D or ...

    if (is2D) {
        parsePool2D(model, layer, input, output);
    } else {
        parsePoolND(model, layer, input, output);
    }
}

}  // namespace vpu
