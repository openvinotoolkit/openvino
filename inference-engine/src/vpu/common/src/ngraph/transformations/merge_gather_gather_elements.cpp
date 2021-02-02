// Copyright (C) 2021 Intel Corporation
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

NGRAPH_RTTI_DEFINITION(vpu::MergeGatherGatherElements, "MergeGatherGatherElements", 0);

namespace vpu {

bool MergeGatherGatherElements::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool wasGraphChanged = false;

    const auto gatherData = ngraph::pattern::any_input();
    const auto gatherIndices = ngraph::pattern::any_input();
    const auto gather = ngraph::pattern::wrap_type<ngraph::opset6::Gather>(
        {gatherData, gatherIndices, ngraph::pattern::any_input()},
        ngraph::pattern::consumers_count(1));

    const auto squeezeAxes = ngraph::pattern::any_input(
        ngraph::pattern::op::as_value_predicate(ngraph::pattern::has_class<ngraph::opset6::Constant>()));
    const auto squeeze = ngraph::pattern::wrap_type<ngraph::opset6::Squeeze>({gather, squeezeAxes}, ngraph::pattern::consumers_count(1));

    const auto transposePerm = ngraph::pattern::any_input(
        ngraph::pattern::op::as_value_predicate(ngraph::pattern::has_class<ngraph::opset6::Constant>()));
    const auto transpose = ngraph::pattern::wrap_type<ngraph::opset6::Transpose>({squeeze, transposePerm});

    const auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "GatherSqueezeTransposeMatcher");
    for (const auto& node : f->get_ordered_ops()) {
        if (m->match(node)) {
            auto& patternMap = m->get_pattern_value_map();

            std::vector<ngraph::Node*> shapeOfs;
            std::vector<ngraph::Node*> gatherElements;

            const auto& m_transpose = patternMap.at(transpose);
            const auto& transposeConsumers = m_transpose.get_target_inputs();
            for (const auto& transposeConsumer : transposeConsumers) {
                if (transposeConsumer.get_node()->get_type_info() == ngraph::opset6::ShapeOf::type_info) {
                    shapeOfs.push_back(transposeConsumer.get_node());
                } else if (transposeConsumer.get_node()->get_type_info() == ngraph::opset6::GatherElements::type_info) {
                    gatherElements.push_back(transposeConsumer.get_node());
                }
            }
            if (gatherElements.empty() || shapeOfs.size() + gatherElements.size() != transposeConsumers.size()) {
                continue;
            }

            const auto& m_transposePerm = patternMap.at(transposePerm);
            const auto& m_squeezeAxes = patternMap.at(squeezeAxes);
            const auto& m_squeeze = patternMap.at(squeeze);
            const auto& m_gatherData = patternMap.at(gatherData);
            const auto& m_gatherIndices = patternMap.at(gatherIndices);
            const auto& m_gather = patternMap.at(gather);

            for (const auto& gatherElement : gatherElements) {
                const auto transposeIndices = std::make_shared<ngraph::opset6::Transpose>(gatherElement->input_value(1), m_transposePerm);
                const auto unsqueezeIndices = std::make_shared<ngraph::opset6::Unsqueeze>(
                    transposeIndices,
                    m_squeezeAxes);
                const auto expGatherElements = std::make_shared<ngraph::vpu::op::ExpGatherElements>(
                    m_gatherData,
                    unsqueezeIndices,
                    m_gatherIndices,
                    ngraph::as_type<ngraph::opset6::GatherElements>(gatherElement)->get_axis(),
                    ngraph::as_type<ngraph::opset6::Gather>(m_gather.get_node())->get_axis());
                const auto squeezeData = m_squeeze.get_node()->clone_with_new_inputs({expGatherElements, m_squeezeAxes});
                const auto transposeData = m_transpose.get_node()->clone_with_new_inputs({squeezeData, m_transposePerm});
                transposeData->set_friendly_name(gatherElement->get_friendly_name());
                gatherElement->get_default_output().replace(transposeData);
                wasGraphChanged = true;
            }

            for (const auto& shapeOf : shapeOfs) {
                const auto gatherDataShape = std::make_shared<ngraph::opset6::ShapeOf>(
                    m_gatherData,
                    ngraph::as_type<ngraph::opset6::ShapeOf>(shapeOf)->get_output_type());
                const auto gatherIndicesShape = std::make_shared<ngraph::opset6::ShapeOf>(
                    m_gatherIndices,
                    ngraph::as_type<ngraph::opset6::ShapeOf>(shapeOf)->get_output_type());
                const auto axis = ngraph::as_type<ngraph::opset6::Gather>(m_gather.get_node())->get_axis();

                const auto gatherIndicesRank = m_gatherIndices.get_partial_shape().rank().get_length();
                const auto gatherDataRank = m_gatherData.get_partial_shape().rank().get_length();
                const auto gatherOutRank = gatherDataRank + gatherIndicesRank - 1;

                std::vector<int64_t> squeezeOutIndices(gatherOutRank);
                std::iota(squeezeOutIndices.begin(), squeezeOutIndices.end(), 0);

                auto squeezeAxes = ngraph::as_type<ngraph::opset6::Constant>(m_squeezeAxes.get_node())->cast_vector<int64_t>();
                ngraph::normalize_axes(m_squeezeAxes.get_node()->description(), squeezeAxes, gatherOutRank);
                std::sort(squeezeAxes.begin(), squeezeAxes.end(), [](int64_t a, int64_t b) { return a > b; });
                for (const auto& squeezeAxis : squeezeAxes) {
                    squeezeOutIndices.erase(squeezeOutIndices.begin() + squeezeAxis);
                }

                std::vector<int64_t> transposeOutIndices(squeezeOutIndices.size());
                auto transposePerm = ngraph::as_type<ngraph::opset6::Constant>(m_transposePerm.get_node())->cast_vector<int64_t>();
                for (size_t i = 0; i < transposeOutIndices.size(); i++) {
                    transposeOutIndices[i] = squeezeOutIndices[transposePerm[i]];
                }

                ngraph::OutputVector gatherOutDims;
                if (axis) {
                    gatherOutDims.push_back(gatherShapeElements(gatherDataShape, 0, axis));
                }
                if (m_gatherIndices.get_partial_shape().rank().get_length()) {
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
                shapeOf->get_default_output().replace(transposeOutShape);
                wasGraphChanged = true;
            }

            m->clear_state();
        }
    }

    return wasGraphChanged;
}

}  // namespace vpu
