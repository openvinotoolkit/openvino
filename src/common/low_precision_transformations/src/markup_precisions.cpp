// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_precisions.hpp"

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

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
        const bool supported = ov::is_type<op::v0::Result>(node) || isSupported(node);
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
        { name<op::v0::Concat>() },
        { name<op::v0::DepthToSpace>() },
        { name<op::v0::Interpolate>() },
        { name<op::v1::MaxPool>() },
        { name<op::v1::ReduceMax>() },
        { name<op::v1::ReduceMin>() },
        { name<op::v0::Relu>() },
        // TODO: there are conditions
        { name<op::v1::BatchToSpace>() },
        { name<op::v1::Broadcast>() },
        { name<op::v3::Broadcast>() },
        { name<op::v1::Pad>() },
        { name<ov::op::v12::Pad>() },
        { name<op::v1::Reshape>() },
        { name<op::v8::Slice>() },
        { name<op::v0::Squeeze>() },
        { name<op::v1::SpaceToBatch>() },
        { name<op::v1::Split>() },
        { name<op::v1::StridedSlice>() },
        { name<op::v0::ShuffleChannels>() },
        { name<op::v1::Transpose>() },
        { name<op::v0::Unsqueeze>() },
        { name<op::v1::VariadicSplit>() },
        { name<op::v1::Gather>() },
        { name<op::v7::Gather>() },
        { name<op::v8::Gather>() },
    };

    const bool precisionPreserved = precisionPreservedOps.find(node->get_type_name()) != precisionPreservedOps.end();
    if (precisionPreserved) {
        return precisionPreserved;
    }

    if (ov::is_type<op::v0::Interpolate>(node)) {
        std::shared_ptr<op::v0::Interpolate> interpolate1 = ov::as_type_ptr<op::v0::Interpolate>(node);
        if (interpolate1) {
            const auto attrs = interpolate1->get_attrs();
            return attrs.mode == "nearest";
        }

        std::shared_ptr<op::v4::Interpolate> interpolate4 = ov::as_type_ptr<op::v4::Interpolate>(node);
        if (interpolate4) {
            const auto attrs = interpolate4->get_attrs();
            return attrs.mode == op::v4::Interpolate::InterpolateMode::NEAREST;
        }
    }

    return false;
}

bool ov::pass::low_precision::MarkupPrecisions::isSupported(const std::shared_ptr<Node>& node) {
    static std::unordered_set<std::string> supportedOps = {
        { name<op::v1::Add>() },
        { name<op::v1::AvgPool>() },
        { name<op::v1::BatchToSpace>() },
        { name<op::v1::Broadcast>() },
        { name<op::v3::Broadcast>() },
        { name<op::v0::Clamp>() },
        { name<op::v0::Concat>() },
        // ?
        { name<op::v0::Convert>() },
        { name<op::v1::Convolution>() },
        { name<op::v1::ConvolutionBackpropData>() },
        { name<op::v0::DepthToSpace>() },
        { name<op::v0::FakeQuantize>() },
        { name<op::v0::Interpolate>() },
        { name<op::v4::Interpolate>() },
        { name<op::v1::GroupConvolution>() },
        { name<op::v0::MatMul>() },
        { name<op::v1::MaxPool>() },
        { name<op::v1::Multiply>() },
        { name<ov::op::v0::MVN>() },
        { name<op::v6::MVN>() },
        { name<op::v0::NormalizeL2>() },
        { name<op::v1::Pad>() },
        { name<ov::op::v12::Pad>() },
        { name<op::v0::PRelu>() },
        { name<op::v1::ReduceMax>() },
        { name<op::v1::ReduceMean>() },
        { name<op::v1::ReduceMin>() },
        { name<op::v1::ReduceSum>() },
        { name<op::v0::Relu>() },
        // TODO: there are conditions
        { name<op::v1::Reshape>() },
        { name<op::v8::Slice>() },
        { name<op::v1::SpaceToBatch>() },
        { name<op::v0::Squeeze>() },
        { name<op::v0::ShuffleChannels>() },
        { name<op::v1::Split>() },
        { name<op::v1::StridedSlice>() },
        // ?
        { name<op::v1::Subtract>() },
        { name<op::v1::Transpose>() },
        { name<op::v0::Unsqueeze>() },
        { name<op::v1::VariadicSplit>() },
        { name<op::v5::LSTMSequence>() },
        { name<op::v5::GRUSequence>() },
        { name<op::v1::Gather>() },
        { name<op::v7::Gather>() },
        { name<op::v8::Gather>() },
    };

    return supportedOps.find(node->get_type_name()) != supportedOps.end();
}
