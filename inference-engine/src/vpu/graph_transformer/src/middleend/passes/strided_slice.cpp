// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

struct StridedSliceParams {
    DimValues begin;
    DimValues end;
    DimValues strides;

    DimValues begin_mask;
    DimValues end_mask;
};

struct StridedSliceInternalParams {
    DimValues begin_dms;
    DimValues end_dms;
    DimValues strides_dms;
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;

    static StridedSliceParams parseInputParams(const Stage& stage);
    static StridedSliceInternalParams computeInternalParams(const Stage& stage, StridedSliceParams params);
};

StridedSliceParams PassImpl::parseInputParams(const Stage& stage) {
    const auto beginInput   = stage->input(1);
    const auto endInput     = stage->input(2);
    const auto num_input_dims = stage->input(0)->desc().numDims();
    StridedSliceParams params;

    IE_ASSERT(beginInput->content() != nullptr);
    IE_ASSERT(endInput->content() != nullptr);

    auto vectorToDimValues = [](const std::vector<int>& v) {
        auto dims = DimsOrder::fromNumDims(v.size()).toIndices();
        int idx = v.size();
        for (auto& dim : dims) {
            idx--;
            dim.second = v[idx];
        }
        return dims;
    };

    params.begin = vectorToDimValues(
        std::vector<int>(beginInput->content()->get<int>(),
                         beginInput->content()->get<int>() + beginInput->desc().dims().get(Dim::C, 0)));
    params.end = vectorToDimValues(
        std::vector<int>(endInput->content()->get<int>(),
                         endInput->content()->get<int>() + endInput->desc().dims().get(Dim::C, 0)));

    // Parse strides input data if needed or set it to default values
    if (stage->numInputs() == 4) {
        const auto stridesInput = stage->input(3);
        IE_ASSERT(stridesInput->content() != nullptr);
        params.strides = vectorToDimValues(
            std::vector<int>(stridesInput->content()->get<int>(),
                             stridesInput->content()->get<int>() + stridesInput->desc().dims().get(Dim::C, 0)));
    } else {
        params.strides = vectorToDimValues(std::vector<int>(num_input_dims, 1));
    }

    IE_ASSERT(params.begin.size() == num_input_dims);
    IE_ASSERT(params.end.size() == num_input_dims);
    IE_ASSERT(params.strides.size() == num_input_dims);

    std::vector<int> begin_mask_values;
    std::vector<int> end_mask_values;

    std::string begin_mask_str = stage->origLayer()->GetParamAsString("begin_mask", "");
    for (const auto& c : begin_mask_str) {
        if (c == '1') begin_mask_values.push_back(1);
        else if (c == '0') begin_mask_values.push_back(0);
    }
    begin_mask_values.insert(begin_mask_values.end(), num_input_dims - begin_mask_values.size(), 1);

    std::string end_mask_str = stage->origLayer()->GetParamAsString("end_mask", "");
    for (const auto& c : end_mask_str) {
        if (c == '1') end_mask_values.push_back(1);
        else if (c == '0') end_mask_values.push_back(0);
    }
    end_mask_values.insert(end_mask_values.end(), num_input_dims - end_mask_values.size(), 1);

    std::string ellipsis_mask_str = stage->origLayer()->GetParamAsString("ellipsis_mask", "");
    for (const auto& c : ellipsis_mask_str) {
        IE_ASSERT(c != '1') << "VPU doesn't support ellipsis_mask for StridedSlice";
    }

    std::string new_axis_mask_str = stage->origLayer()->GetParamAsString("new_axis_mask", "");
    for (const auto& c : new_axis_mask_str) {
        IE_ASSERT(c != '1') << "VPU doesn't support new_axis_mask for StridedSlice";
    }

    std::string shrink_axis_mask_str = stage->origLayer()->GetParamAsString("shrink_axis_mask", "");
    for (const auto& c : shrink_axis_mask_str) {
        IE_ASSERT(c != '1') << "VPU doesn't support shrink_axis_mask for StridedSlice";
    }

    params.begin_mask = vectorToDimValues(begin_mask_values);
    params.end_mask = vectorToDimValues(end_mask_values);

    return params;
}

StridedSliceInternalParams PassImpl::computeInternalParams(const Stage& stage, StridedSliceParams params) {
    auto input = stage->input(0);

    StridedSliceInternalParams m_params = StridedSliceInternalParams();
    size_t numDims = input->desc().numDims();

    for (const auto&  dim : input->desc().dimsOrder().toPermutation()) {
        m_params.begin_dms.set(dim, 0);
        m_params.end_dms.set(dim, input->desc().dim(dim));
        m_params.strides_dms.set(dim, 1);
    }

    auto clip = [](int value, int min, int max) {
        return std::min(std::max(min, value), max);
    };

    for (const auto& dim : input->desc().dimsOrder().toPermutation()) {
        m_params.strides_dms.set(dim, params.strides[dim]);

        IE_ASSERT(params.begin_mask[dim] == 1 || params.begin_mask[dim] == 0);
        IE_ASSERT(params.end_mask[dim] == 1 || params.end_mask[dim] == 0);

        m_params.begin_dms.set(dim,
            params.begin_mask[dim] ? clip(params.begin[dim], 0, input->desc().dim(dim)) : 0);
        m_params.end_dms.set(dim,
            params.end_mask[dim] ? clip(params.end[dim], 0, input->desc().dim(dim)) : input->desc().dim(dim));

        IE_ASSERT(dim != Dim::N || numDims < 4 || m_params.strides_dms[dim] == 1)
            << "VPU doesn't support batch strides for StridedSlice";
        IE_ASSERT(m_params.begin_dms[dim] >= 0 && m_params.begin_dms[dim] < m_params.end_dms[dim]);
        IE_ASSERT(m_params.end_dms[dim] <= input->desc().dim(dim));
        IE_ASSERT(m_params.strides_dms[dim] > 0);
    }

    return m_params;
}

void PassImpl::run(const Model& model) {
    VPU_PROFILE(stridedSlice);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StridedSlice) {
            continue;
        }
        IE_ASSERT(stage->numInputs() == 3 || stage->numInputs() == 4);
        IE_ASSERT(stage->numOutputs() == 1);

        auto input  = stage->input(0);
        auto output = stage->output(0);

        IE_ASSERT(input->desc().numDims() == output->desc().numDims());

        auto params = parseInputParams(stage);
        auto m_params = computeInternalParams(stage, params);

        model->disconnectStage(stage);

        auto directOrder = DimsOrder::fromNumDims(input->desc().numDims());
        auto perm = directOrder.toPermutation();

        //
        // Select a region of interest in accordance with the begin and end parameters.
        //

        const bool needSelectROI = std::any_of(perm.begin(), perm.end(), [&](Dim dim) {
            return m_params.begin_dms[dim] != 0 || m_params.end_dms[dim] != input->desc().dim(dim); });
        if (needSelectROI) {
            auto roiDesc = input->desc();
            for (const auto &dim : perm) {
                roiDesc.setDim(dim, m_params.end_dms[dim] - m_params.begin_dms[dim]);
            }
            auto roiData = model->duplicateData(input, "@roi", roiDesc);
            auto shrinkStage = _stageBuilder->addShrinkStage(
                model,
                stage->name() + "@roi-selection",
                stage->origLayer(),
                input,
                roiData);
            shrinkStage->attrs().set("offset", m_params.begin_dms);
            input = roiData;
        }

        //
        // Expand each dimension of the input tensor, if it is not completely divided by stride
        // for further work.
        //

        const bool needExpand = std::any_of(perm.begin(), perm.end(), [&](Dim dim) {
            return input->desc().dim(dim) % m_params.strides_dms[dim] != 0; });
        if (needExpand) {
            auto expandDesc = input->desc();
            for (const auto& dim : perm) {
                auto alignValue = (m_params.strides_dms[dim] - expandDesc.dim(dim) % m_params.strides_dms[dim])
                    % m_params.strides_dms[dim];
                expandDesc.setDim(dim, expandDesc.dim(dim) + alignValue);
            }
            auto expandedInputData = model->duplicateData(input, "@extended-input", expandDesc);
            _stageBuilder->addExpandStage(
                model,
                stage->name() + "@expand-input",
                stage->origLayer(),
                input,
                expandedInputData);
            input = expandedInputData;
        }

        //
        // For copying with stride we do reshape in order to put data of interest at the beginning of each dimension,
        // split into necessary and unnecessary data and then reverse reshape.
        //

        for (const auto& dim : perm) {
            if (m_params.strides_dms[dim] == 1)
                continue;

            auto stride = abs(m_params.strides_dms[dim]);
            auto reshapedDesc = input->desc();
            auto subtensorDesc = input->desc();
            auto intermediateOutDesc = input->desc();

            if (input->desc().numDims() == 1) {
                reshapedDesc = DataDesc({stride, input->desc().dim(dim) / stride});
                subtensorDesc = DataDesc({1, input->desc().dim(dim) / stride});
            } else if (perm.front() == dim) {
                auto nextDim = perm.at(directOrder.dimInd(dim) + 1);
                reshapedDesc.setDim(dim, stride);
                reshapedDesc.setDim(nextDim,
                                    input->desc().dim(dim) * input->desc().dim(nextDim) / stride);
                subtensorDesc.setDim(dim, 1);
                subtensorDesc.setDim(nextDim, reshapedDesc.dim(nextDim));
            } else {
                auto previousDim = perm.at(directOrder.dimInd(dim) - 1);
                reshapedDesc.setDim(dim, input->desc().dim(dim) / stride);
                reshapedDesc.setDim(previousDim, input->desc().dim(previousDim) * stride);

                subtensorDesc.setDim(dim, reshapedDesc.dim(dim));
                subtensorDesc.setDim(previousDim, input->desc().dim(previousDim));
            }

            intermediateOutDesc.setDim(dim, input->desc().dim(dim) / stride);

            auto reshapedInputData = model->duplicateData(
                input, formatString("@reshaped-input@dim%s", dim), reshapedDesc);
            auto subtensorData = model->duplicateData(
                input, formatString("@subtensor@dim%s", dim), subtensorDesc);
            auto intermediateOutputData = model->duplicateData(
                input, formatString("@intermediate-output@dim%s", dim), intermediateOutDesc);

            _stageBuilder->addReshapeStage(
                model,
                formatString("%s@reshape-input@dim%s", stage->name(), dim),
                stage->origLayer(),
                input,
                reshapedInputData);

            _stageBuilder->addSplitStage(
                model,
                formatString("%s@split@dim%s", stage->name(), dim),
                stage->origLayer(),
                dim,
                reshapedInputData,
                {subtensorData});

            _stageBuilder->addReshapeStage(
                model,
                formatString("%s@reshape-output@dim%s", stage->name(), dim),
                stage->origLayer(),
                subtensorData,
                intermediateOutputData);

            input = intermediateOutputData;
        }

        _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-output", stage->name()),
            stage->origLayer(),
            input,
            output,
            "stridedSlice");

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::stridedSlice() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
