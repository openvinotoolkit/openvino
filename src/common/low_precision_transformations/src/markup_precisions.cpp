// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_precisions.hpp"

#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

using namespace ngraph;

ngraph::pass::low_precision::MarkupPrecisions::MarkupPrecisions(
    const std::vector<PrecisionsRestriction>& restrictions,
    const std::vector<ngraph::element::Type>& defaultPrecisions) : defaultPrecisions(defaultPrecisions) {
    for (const auto& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (it == restrictionsByOperation.end()) {
            Restriction r(restriction.specifyVersion);
            if (!restriction.inputPrecisionsByPort.empty()) {
                r.inputPrecisionsByVersion.emplace(restriction.operationType.version, restriction.inputPrecisionsByPort);
            }

            if (!restriction.outputPrecisionsByPort.empty()) {
                r.outputPrecisionsByVersion.emplace(restriction.operationType.version, restriction.outputPrecisionsByPort);
            }

            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            Restriction& r = it->second;
            it->second.add(restriction.operationType.version, restriction.inputPrecisionsByPort, restriction.outputPrecisionsByPort);

            r.inputPrecisionsByVersion.emplace(restriction.operationType.version, restriction.inputPrecisionsByPort);
            r.outputPrecisionsByVersion.emplace(restriction.operationType.version, restriction.outputPrecisionsByPort);
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
}

namespace {
void setRestriction(
    const std::shared_ptr<Node>& node,
    const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisionsByPort,
    const bool setForInput) {
    if (precisionsByPort.empty()) {
        for (auto i = 0; i < (setForInput ? node->get_input_size() : node->get_output_size()); ++i) {
            auto& rt = setForInput ? node->input(i).get_rt_info() : node->output(i).get_rt_info();
            rt.emplace(
                PrecisionsAttribute::get_type_info_static(),
                PrecisionsAttribute(std::vector<element::Type>()));
        }
    } else {
        for (const std::pair<size_t, std::vector<ngraph::element::Type>>& item : precisionsByPort) {
            auto& rt = setForInput ? node->input(item.first).get_rt_info() : node->output(item.first).get_rt_info();

            auto precisionsAttribute = setForInput ?
                ngraph::pass::low_precision::getAttribute<PrecisionsAttribute>(node->input(item.first)) :
                ngraph::pass::low_precision::getAttributeFromOutput<PrecisionsAttribute>(node->output(item.first));
            if ((!precisionsAttribute.empty()) &&
                (precisionsAttribute.as<PrecisionsAttribute>().value().empty())) {
                return;
            }
            rt[PrecisionsAttribute::get_type_info_static()] = PrecisionsAttribute(item.second);
        }
    }
}
} // namespace

bool ngraph::pass::low_precision::MarkupPrecisions::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupPrecisions);
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        if (transformation_callback(node)) {
            continue;
        }

        // TODO: don't need to set restrictions for not supported operations
        // if don't set restrictions for not supported operations then accuracy drop appears, issue #59197
        const bool supported = ov::is_type<opset1::Result>(node) || isSupported(node);
        if (!supported || !LayerTransformation::canBeTransformedStatic(node, defaultPrecisions)) {
            setRestriction(node, std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>> { {0ul, {}}}, true);
            continue;
        }

        const bool precisionPreserved = isPrecisionPreserved(node);
        if (precisionPreserved) {
            auto& rt = node->get_rt_info();
            rt.emplace(
                PrecisionPreservedAttribute::get_type_info_static(),
                PrecisionPreservedAttribute(precisionPreserved));
        }

        const auto& typeInfo = node->get_type_info();
        auto it = restrictionsByOperation.find(typeInfo.name);
        if (it != restrictionsByOperation.end()) {
            const Restriction& r = it->second;
            auto setRestriction = [&](const std::shared_ptr<Node>& node, const Restriction& r, const bool input) {
                const auto& precisionsByVersion = input ? r.inputPrecisionsByVersion : r.outputPrecisionsByVersion;
                if (precisionsByVersion.empty()) {
                    return;
                }

                if (r.versionIsRequired) {
                    const auto& typeInfo = node->get_type_info();
                    OPENVINO_SUPPRESS_DEPRECATED_START
                    const auto it2 = precisionsByVersion.find(typeInfo.version);
                    OPENVINO_SUPPRESS_DEPRECATED_END
                    if (it2 == precisionsByVersion.end()) {
                        return;
                    }

                    const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisionsByPort = it2->second;
                    ::setRestriction(node, precisionsByPort, input);
                } else {
                    assert(precisionsByVersion.size() == 1ul);

                    const std::vector<std::pair<size_t, std::vector<ngraph::element::Type>>>& precisionsByPort = precisionsByVersion.begin()->second;
                    ::setRestriction(node, precisionsByPort, input);
                }
            };
            setRestriction(node, r, true);
            setRestriction(node, r, false);
        }
    }
    return true;
}

template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
    if (isDisabled(node)) {
        return false;
    }

    // TODO: think how to handle conditions <= not mandatory for PoC
    // TODO: operation set version is not affected <= not mandatory for PoC
    static std::unordered_set<std::string> precisionPreservedOps = {
        { name<opset1::Concat>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::Interpolate>() },
        { name<opset1::MaxPool>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Pad>() },
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() }
    };

    const bool precisionPreserved = precisionPreservedOps.find(node->get_type_name()) != precisionPreservedOps.end();
    if (precisionPreserved) {
        return precisionPreserved;
    }

    if (ov::is_type<opset1::Interpolate>(node)) {
        std::shared_ptr<opset1::Interpolate> interpolate1 = ov::as_type_ptr<opset1::Interpolate>(node);
        if (interpolate1) {
            const auto attrs = interpolate1->get_attrs();
            return attrs.mode == "nearest";
        }

        std::shared_ptr<opset4::Interpolate> interpolate4 = ov::as_type_ptr<opset4::Interpolate>(node);
        if (interpolate4) {
            const auto attrs = interpolate4->get_attrs();
            return attrs.mode == op::v4::Interpolate::InterpolateMode::NEAREST;
        }
    }

    return false;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isSupported(const std::shared_ptr<Node>& node) {
    static std::unordered_set<std::string> supportedOps = {
        { name<opset1::Add>() },
        { name<opset1::AvgPool>() },
        { name<opset1::Clamp>() },
        { name<opset1::Concat>() },
        // ?
        { name<opset1::Convert>() },
        { name<opset1::Convolution>() },
        { name<opset1::ConvolutionBackpropData>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::FakeQuantize>() },
        { name<opset1::Interpolate>() },
        { name<opset4::Interpolate>() },
        { name<opset1::GroupConvolution>() },
        { name<opset1::MatMul>() },
        { name<opset1::MaxPool>() },
        { name<opset1::Multiply>() },
        { name<ngraph::op::MVN>() },
        { name<opset6::MVN>() },
        { name<opset1::NormalizeL2>() },
        { name<opset1::Pad>() },
        { name<opset1::PRelu>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMean>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::ReduceSum>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        // ?
        { name<opset1::Subtract>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() }
    };

    return supportedOps.find(node->get_type_name()) != supportedOps.end();
}
