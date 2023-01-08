// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/merge_gather_gather_elements.hpp"

#include <vpu/ngraph/operations/exp_gather_elements.hpp>
#include <vpu/ngraph/utilities.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/log.hpp>

#include <numeric>

namespace vpu {

bool MergeGatherGatherElements::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    bool wasGraphChanged = false;

    const auto gatherData = ngraph::pattern::any_input();
    const auto gatherIndices = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    const auto gatherAxis = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    const auto gather = ngraph::pattern::wrap_type<ngraph::opset6::Gather>(
        {gatherData, gatherIndices, gatherAxis},
        ngraph::pattern::consumers_count(1));

    const auto squeezeAxes = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    const auto squeeze = ngraph::pattern::wrap_type<ngraph::opset6::Squeeze>({gather, squeezeAxes}, ngraph::pattern::consumers_count(1));

    const auto transposePerm = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    const auto transpose = ngraph::pattern::wrap_type<ngraph::opset6::Transpose>({squeeze, transposePerm});

    const auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "GatherSqueezeTransposeMatcher");
    for (const auto& node : f->get_ordered_ops()) {
        if (!m->match(node)) {
            continue;
        }

        const auto& patternMap = m->get_pattern_value_map();

        std::vector<ngraph::Node*> shapeOfs;
        std::vector<ngraph::Node*> gatherElements;

        const auto& matchedTranspose = patternMap.at(transpose);
        const auto& transposeConsumers = matchedTranspose.get_target_inputs();

        const auto isShapeOfOrGatherElements = [](const ngraph::Input<ngraph::Node>& consumer) {
            return consumer.get_node()->get_type_info().is_castable(ngraph::opset6::ShapeOf::get_type_info_static()) ||
                   consumer.get_node()->get_type_info().is_castable(ngraph::opset6::GatherElements::get_type_info_static());
        };

        if (!std::all_of(transposeConsumers.cbegin(), transposeConsumers.cend(), isShapeOfOrGatherElements)) {
            continue;
        }

        for (const auto& transposeConsumer : transposeConsumers) {
            if (transposeConsumer.get_node()->get_type_info().is_castable(ngraph::opset6::ShapeOf::get_type_info_static())) {
                shapeOfs.push_back(transposeConsumer.get_node());
            } else if (transposeConsumer.get_node()->get_type_info().is_castable(ngraph::opset6::GatherElements::get_type_info_static())) {
                gatherElements.push_back(transposeConsumer.get_node());
            }
        }

        const auto& matchedTransposePerm = patternMap.at(transposePerm);
        const auto& matchedSqueezeAxes = patternMap.at(squeezeAxes);
        const auto& matchedSqueeze = patternMap.at(squeeze);
        const auto& matchedGatherData = patternMap.at(gatherData);
        const auto& matchedGatherIndices = patternMap.at(gatherIndices);
        const auto& matchedGatherAxis = patternMap.at(gatherAxis);

        const auto axisConst = ngraph::get_constant_from_source(matchedGatherAxis);
        if (!axisConst) {
            continue;
        }
        const auto axisVec = axisConst->cast_vector<int64_t>();
        VPU_THROW_UNLESS(axisVec.size() == 1, "Error while merging Gather and GatherElements: expected to get one value from axis input, but got: {}",
                         axisVec.size());
        const auto axis = axisVec.front();

        for (const auto& gatherElement : gatherElements) {
            const auto transposeIndices = std::make_shared<ngraph::opset6::Transpose>(gatherElement->input_value(1), matchedTransposePerm);
            const auto unsqueezeIndices = std::make_shared<ngraph::opset6::Unsqueeze>(
                transposeIndices,
                matchedSqueezeAxes);
            const auto expGatherElements = std::make_shared<ngraph::vpu::op::ExpGatherElements>(
                matchedGatherData,
                unsqueezeIndices,
                matchedGatherIndices,
                ngraph::as_type<ngraph::opset6::GatherElements>(gatherElement)->get_axis(),
                axis);
            const auto squeezeData = matchedSqueeze.get_node()->clone_with_new_inputs({expGatherElements, matchedSqueezeAxes});
            const auto transposeData = matchedTranspose.get_node()->clone_with_new_inputs({squeezeData, matchedTransposePerm});
            transposeData->set_friendly_name(gatherElement->get_friendly_name());
            gatherElement->output(0).replace(transposeData);
            wasGraphChanged = true;
        }

        for (const auto& shapeOf : shapeOfs) {
            const auto gatherDataShape = std::make_shared<ngraph::opset6::ShapeOf>(
                matchedGatherData,
                ngraph::as_type<ngraph::opset6::ShapeOf>(shapeOf)->get_output_type());
            const auto gatherIndicesShape = std::make_shared<ngraph::opset6::ShapeOf>(
                matchedGatherIndices,
                ngraph::as_type<ngraph::opset6::ShapeOf>(shapeOf)->get_output_type());

            const auto gatherIndicesRank = matchedGatherIndices.get_partial_shape().rank().get_length();
            const auto gatherDataRank = matchedGatherData.get_partial_shape().rank().get_length();
            const auto gatherOutRank = gatherDataRank + gatherIndicesRank - 1;

            std::vector<int64_t> squeezeOutIndices(gatherOutRank);
            std::iota(squeezeOutIndices.begin(), squeezeOutIndices.end(), 0);

            const auto normedSqueezeAxes = ngraph::normalize_axes(
                matchedSqueezeAxes.get_node()->description(),
                ngraph::as_type<ngraph::opset6::Constant>(matchedSqueezeAxes.get_node())->cast_vector<int64_t>(),
                gatherOutRank);
            const std::set<size_t, std::greater<size_t>> orderedSqueezeAxes(normedSqueezeAxes.begin(), normedSqueezeAxes.end());
            for (const auto& squeezeAxis : orderedSqueezeAxes) {
                squeezeOutIndices.erase(squeezeOutIndices.begin() + squeezeAxis);
            }

            std::vector<int64_t> transposeOutIndices(squeezeOutIndices.size());
            auto transposePerm = ngraph::as_type<ngraph::opset6::Constant>(matchedTransposePerm.get_node())->cast_vector<int64_t>();
            for (size_t i = 0; i < transposeOutIndices.size(); i++) {
                transposeOutIndices[i] = squeezeOutIndices[transposePerm[i]];
            }

            ngraph::OutputVector gatherOutDims;
            if (axis) {
                gatherOutDims.push_back(gatherShapeElements(gatherDataShape, 0, axis));
            }
            if (matchedGatherIndices.get_partial_shape().rank().get_length()) {
                gatherOutDims.push_back(gatherIndicesShape);
            }
            if (axis + 1 < gatherDataRank) {
                gatherOutDims.push_back(gatherShapeElements(gatherDataShape, axis + 1, gatherDataRank - axis - 1));
            }

            const auto gatherOutShape = std::make_shared<ngraph::opset6::Concat>(gatherOutDims, 0);

            const auto transposeOutShape = std::make_shared<ngraph::opset6::Gather>(
                gatherOutShape,
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{transposeOutIndices.size()}, transposeOutIndices),
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));

            transposeOutShape->set_friendly_name(shapeOf->get_friendly_name());
            shapeOf->output(0).replace(transposeOutShape);
            wasGraphChanged = true;
        }
    }

    return wasGraphChanged;
}

}  // namespace vpu
