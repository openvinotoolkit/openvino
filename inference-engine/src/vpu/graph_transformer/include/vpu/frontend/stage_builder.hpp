// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_layers.h>

#include <vpu/model/model.hpp>

namespace vpu {

class StageBuilder final : public std::enable_shared_from_this<StageBuilder> {
public:
    using Ptr = std::shared_ptr<StageBuilder>;

    Stage createConvertStage(
            const Model::Ptr& model,
            const std::string& name,
            const Data& input,
            const Data& output,
            StageType type,
            float scale = 1.0f,
            float bias = 0.0f);

    Stage addSumStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input0,
            const Data& input1,
            const Data& output);

    Stage addBiasStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& biases,
            const Data& output);

    Stage addScaleStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& scales,
            const Data& output);

    Stage addCopyStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output);

    Stage addPadStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            PadMode padMode,
            float pad_value,
            const DimValues& pads_begin,
            const DimValues& pads_end,
            const Data& input,
            const Data& output);

    Stage addNoneStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const DataVector& inputs,
            const DataVector& outputs);

    Stage addPowerStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float scale,
            float power,
            float bias,
            const Data& input,
            const Data& output);

    Stage addReLUStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float negativeSlope,
            const Data& input,
            const Data& output,
            const Data& biases = nullptr);

    Stage addReshapeStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output);

    Stage addConcatStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            Dim axis,
            const DataVector& inputs,
            const Data& output);

    Stage addConcatStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const std::vector<DimValues>& offsets,
            const DataVector& inputs,
            const Data& output);

    Stage addSplitStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            Dim axis,
            const Data& input,
            const DataVector& outputs);

    Stage addSplitStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const std::vector<DimValues>& offsets,
            const Data& input,
            const DataVector& outputs);

    Stage addScalingStage(
            const Model::Ptr& model,
            const ie::CNNLayerPtr& origLayer,
            float scale,
            const Data& input,
            const Data& output);

    Stage addSwFullyConnectedStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& weights,
            const Data& biases,
            Data output);

    Stage addExpandStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const DimValues& offset = DimValues());

    Stage addShrinkStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const DimValues& offset = DimValues());

    Stage addSoftMaxStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            Dim axis);

    Stage addClampStage(
            const Model::Ptr& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float min,
            float max,
            const Data& input,
            const Data& output);
};

}  // namespace vpu
