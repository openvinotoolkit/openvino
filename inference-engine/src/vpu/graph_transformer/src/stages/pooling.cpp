// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

#include <ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>

namespace vpu {

void FrontEnd::parsePooling(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    if (!(input->desc().numDims() == 3 || input->desc().numDims() == 4)) {
        VPU_THROW_EXCEPTION << "Pooling supports only 3D or 4D input";
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_EXCEPTION << "Pooling supports only same num dims in input and output";
    }

    //
    // Extract parameters
    //

    auto poolLayer = std::dynamic_pointer_cast<ie::PoolingLayer>(layer);
    IE_ASSERT(poolLayer != nullptr);

    int kernelSizeX = poolLayer->_kernel_x;
    int kernelSizeY = poolLayer->_kernel_y;

    int kernelStrideX = poolLayer->_stride_x;
    int kernelStrideY = poolLayer->_stride_y;

    auto paddings = getPaddings(*poolLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    auto poolType = poolLayer->_type;

    auto excludePad = poolLayer->_exclude_pad;

    //
    // Check if HW is applicable
    //

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

        int outWidthWithOutCeil = (tempX + kernelStrideX) / kernelStrideX;
        int outHeightWithOutCeil = (tempY + kernelStrideX) / kernelStrideX;

        int outWidthWithCeil =  static_cast<int>(std::ceil(static_cast<double>(tempX) / kernelStrideX + 1));
        int outHeightWithCeil = static_cast<int>(std::ceil(static_cast<double>(tempY) / kernelStrideX + 1));

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

    //  FIX #14949, enable HW AVG pooling, need SW postproc
    if (excludePad && poolType == ie::PoolingLayer::PoolType::AVG) {
        if (output->desc().dim(Dim::W) == 5 &&
            output->desc().dim(Dim::H) == 5 &&
            kernelSizeX == 5 &&
            kernelSizeY == 5) {
            tryHW = false;
        }
    }


    if (env.netConfig.hwDisabled(layer->name)) {
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

}  // namespace vpu
