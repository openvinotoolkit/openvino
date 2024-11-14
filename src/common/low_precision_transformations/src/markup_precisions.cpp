// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_precisions.hpp"

#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

using namespace ov;

ov::pass::low_precision::MarkupPrecisions::MarkupPrecisions(
    const std::vector<PrecisionsRestriction>& restrictions,
    const std::vector<ov::element::Type>& defaultPrecisions) : defaultPrecisions(defaultPrecisions) {
    for (const auto& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        if (it == restrictionsByOperation.end()) {
            Restriction r(restriction.specifyVersion);
            r.precisionsByVersion.emplace(
                restriction.operationType.version_id,
                Restriction::RestrictionByVersion(restriction.precisionsByPortsFunction, restriction.precisionsByPorts));
            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            it->second.add(
                restriction.operationType.version_id,
                Restriction::RestrictionByVersion(restriction.precisionsByPortsFunction, restriction.precisionsByPorts));
        }
    }
}

namespace {
void setRestriction(
    const std::shared_ptr<Node>& node,
    const pass::low_precision::PrecisionsRestriction::PrecisionsByPorts& precisionsByPorts) {
    if (precisionsByPorts.empty()) {
        // if available precisions for any port is empty then mark all input ports
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();
            rt.emplace(
                    PrecisionsAttribute::get_type_info_static(),
                    PrecisionsAttribute(std::vector<element::Type>()));
        }
    } else {
        for (const auto& item : precisionsByPorts) {
            const auto attr = PrecisionsAttribute(item.second);
            for (const auto& port : item.first) {
                Input<Node> input = node->input(port);
                auto& rt = input.get_rt_info();
                auto precisionsAttribute = ov::pass::low_precision::getAttribute<PrecisionsAttribute>(input);
                if ((!precisionsAttribute.empty()) && (precisionsAttribute.as<PrecisionsAttribute>().value().empty())) {
                    return;
                }
                rt[PrecisionsAttribute::get_type_info_static()] = attr;
            }
        }
    }
}
} // namespace

bool ov::pass::low_precision::MarkupPrecisions::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupPrecisions);
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        if (transformation_callback(node)) {
            continue;
        }

        if (const auto multiSubGraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++)
                run_on_model(multiSubGraph->get_function(i));
            continue;
        }

        // TODO: don't need to set restrictions for not supported operations
        // if don't set restrictions for not supported operations then accuracy drop appears, issue #59197
        const bool supported = ov::is_type<opset1::Result>(node) || isSupported(node);
        if (!supported && restrictionsByOperation.find(node->get_type_info().name) != restrictionsByOperation.end())
            THROW_IE_LPT_EXCEPTION(*node) << "Restriction is set for unsupported operation";
        if (!supported || !LayerTransformation::canBeTransformedStatic(node, defaultPrecisions)) {
            setRestriction(node, pass::low_precision::PrecisionsRestriction::PrecisionsByPorts{{{0ul}, {}}});
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
            if (r.versionIsRequired) {
                const auto it2 = r.precisionsByVersion.find(typeInfo.version_id);
                if (it2 == r.precisionsByVersion.end()) {
                    continue;
                }

                const auto& precisionsByPorts = it2->second;
                setRestriction(node, precisionsByPorts.get(node));
            } else {
                assert(r.precisionsByVersion.size() == 1ul);

                const auto& precisionsByPorts = r.precisionsByVersion.begin()->second;
                setRestriction(node, precisionsByPorts.get(node));
            }
        }
    }
    return true;
}

template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool ov::pass::low_precision::MarkupPrecisions::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
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
        { name<opset2::BatchToSpace>() },
        { name<opset1::Broadcast>() },
        { name<opset3::Broadcast>() },
        { name<opset1::Pad>() },
        { name<ov::opset12::Pad>() },
        { name<opset1::Reshape>() },
        { name<opset8::Slice>() },
        { name<opset1::Squeeze>() },
        { name<opset2::SpaceToBatch>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() },
        { name<opset1::Gather>() },
        { name<opset7::Gather>() },
        { name<opset8::Gather>() },
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

bool ov::pass::low_precision::MarkupPrecisions::isSupported(const std::shared_ptr<Node>& node) {
    static std::unordered_set<std::string> supportedOps = {
        { name<opset1::Add>() },
        { name<opset1::AvgPool>() },
        { name<opset2::BatchToSpace>() },
        { name<opset1::Broadcast>() },
        { name<opset3::Broadcast>() },
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
        { name<ov::op::v0::MVN>() },
        { name<opset6::MVN>() },
        { name<opset1::NormalizeL2>() },
        { name<opset1::Pad>() },
        { name<ov::opset12::Pad>() },
        { name<opset1::PRelu>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMean>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::ReduceSum>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Reshape>() },
        { name<opset8::Slice>() },
        { name<opset2::SpaceToBatch>() },
        { name<opset1::Squeeze>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        // ?
        { name<opset1::Subtract>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() },
        { name<opset5::LSTMSequence>() },
        { name<opset6::GRUSequence>() },
        { name<opset1::Gather>() },
        { name<opset7::Gather>() },
        { name<opset8::Gather>() },
    };

    return supportedOps.find(node->get_type_name()) != supportedOps.end();
}
