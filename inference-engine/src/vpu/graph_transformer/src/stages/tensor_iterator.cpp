// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/stages/iteration_rule.hpp"
#include "vpu/utils/auto_scope.hpp"
#include "vpu/compile_env.hpp"
#include "graph_transformer.h"
#include "vpu/model/data_contents/ie_blob_content.hpp"

#include "ie_layers_internal.hpp"
#include "net_pass.h"

#include <unordered_map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <map>
#include <string>

namespace vpu {

namespace {

using PortMap = ie::TensorIterator::PortMap;

bool isIterable(const PortMap& rule) {
    return rule.axis != -1;
}

bool isIterableInput(const ie::DataPtr& data, const std::shared_ptr<ie::TensorIterator>& tensorIterator) {
    const auto isInput = [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->body.inputs[rule.to] == data; };
    const auto& rules = tensorIterator->input_port_map;
    return std::any_of(rules.begin(), rules.end(), [&isInput](const PortMap& rule) { return isIterable(rule) && isInput(rule); });
}

bool isIterableOutput(const ie::DataPtr& data, const std::shared_ptr<ie::TensorIterator>& tensorIterator) {
    const auto isOutput = [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->body.outputs[rule.to] == data; };
    const auto& rules = tensorIterator->output_port_map;
    return std::any_of(rules.begin(), rules.end(), [&isOutput](const PortMap& rule) { return isIterable(rule) && isOutput(rule); });
}

bool isIterable(const ie::DataPtr& data, const std::shared_ptr<ie::TensorIterator>& tensorIterator) {
    const auto& bodyInputs = tensorIterator->body.inputs;
    const auto& bodyOutputs = tensorIterator->body.outputs;

    const bool isBodyInput = std::find(bodyInputs.begin(), bodyInputs.end(), data) != bodyInputs.end();
    const bool isBodyOutput = std::find(bodyOutputs.begin(), bodyOutputs.end(), data) != bodyOutputs.end();
    VPU_THROW_UNLESS(isBodyInput || isBodyOutput, "Check on iterable component is valid only for Tensor Iterator's body input and output data objects");

    return isIterableInput(data, tensorIterator) || isIterableOutput(data, tensorIterator);
}

bool hasBackEdgeConnectionTo(const ie::DataPtr& data, const std::shared_ptr<ie::TensorIterator>& tensorIterator) {
    const auto& rules = tensorIterator->back_edges;
    return std::any_of(rules.begin(), rules.end(), [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->body.inputs[rule.to] == data; });
}

bool isConst(const ie::CNNLayerPtr& layer) {
    return layer->type == "Const" && layer->outData.size() == 1 && layer->blobs.size() == 1;
}

bool isConst(const ie::DataPtr& data) {
    const auto creator = data->getCreatorLayer().lock();
    return creator != nullptr && isConst(creator);
}

bool isFakeHolder(const ie::DataPtr& data) {
    return data->getPrecision() == ie::Precision::UNSPECIFIED;
}

}  // namespace

void FrontEnd::parseTensorIterator(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) {
    IE_ASSERT(!inputs.empty());
    IE_ASSERT(!outputs.empty());

    auto tensorIterator = std::dynamic_pointer_cast<ie::TensorIterator>(layer);
    IE_ASSERT(tensorIterator != nullptr);

    auto createDescriptor = [&](const ie::TensorDesc& original) {
        auto vpuDescriptor = DataDesc{original};
        if (vpuDescriptor.type() == DataType::FP32) {
            // to infer the same FP32 models on different devices (CPU, GPU, VPU and so on)
            vpuDescriptor.setType(DataType::FP16);
        }
        return vpuDescriptor;
    };

    auto createData = [&](const ie::DataPtr& original) {
        return model->addNewData(original->getName(), createDescriptor(original->getTensorDesc()));;
    };

    auto createConstData = [&](const ie::DataPtr& original) {
        VPU_THROW_UNLESS(isConst(original), "VPU const data object can be created only from const IE data object");

        const auto& creator = original->getCreatorLayer().lock();
        const auto& descriptor = createDescriptor(original->getTensorDesc());
        const auto& blob = ieBlobContent(creator->blobs.begin()->second, descriptor.type());

        return model->addConstData(original->getName(), descriptor, blob);
    };

    auto findTIInputsDataByBodyData = [&](const ie::DataPtr& bodyData) -> std::vector<ie::DataPtr> {
        std::vector<ie::DataPtr> tensorIteratorInputs;
        for (const auto& rule : tensorIterator->input_port_map) {
            if (tensorIterator->body.inputs[rule.to] == bodyData) {
                tensorIteratorInputs.push_back(tensorIterator->insData[rule.from].lock());
            }
        }
        return tensorIteratorInputs;
    };

    auto findTIOutputsDataByBodyData = [&](const ie::DataPtr& bodyData) -> std::vector<ie::DataPtr> {
        std::vector<ie::DataPtr> tensorIteratorOutputs;
        for (const auto& rule : tensorIterator->output_port_map) {
            if (tensorIterator->body.outputs[rule.to] == bodyData) {
                tensorIteratorOutputs.push_back(tensorIterator->outData[rule.from]);
            }
        }
        return tensorIteratorOutputs;
    };

    auto getBodyOutputsByBodyInput = [&](const ie::DataPtr& bodyInput) -> std::vector<ie::DataPtr> {
        std::vector<ie::DataPtr> bodyOutputs;
        for (const auto& rule : tensorIterator->back_edges) {
            if (tensorIterator->body.inputs[rule.to] == bodyInput) {
                bodyOutputs.push_back(tensorIterator->body.outputs[rule.from]);
            }
        }
        return bodyOutputs;
    };

    auto getInputIterableRule = [&](const ie::DataPtr& from, const ie::DataPtr& to) {
        std::vector<PortMap> rules;
        for (const auto& rule : tensorIterator->input_port_map) {
            if (isIterable(rule) && tensorIterator->insData[rule.from].lock() == from && tensorIterator->body.inputs[rule.to] == to) {
                rules.push_back(rule);
            }
        }
        VPU_THROW_UNLESS(!rules.empty(), "There must be an iterable rule between data objects");
        VPU_THROW_UNLESS(rules.size() == 1, "There cannot be more than one iterable rule with the same source and destination");
        return rules.front();
    };

    auto getOutputIterableRule = [&](const ie::DataPtr& from, const ie::DataPtr& to) {
        std::vector<PortMap> rules;
        for (const auto& rule : tensorIterator->output_port_map) {
            if (isIterable(rule) && tensorIterator->outData[rule.from] == from && tensorIterator->body.outputs[rule.to] == to) {
                rules.push_back(rule);
            }
        }
        VPU_THROW_UNLESS(!rules.empty(), "There must be an iterable rule between data objects");
        VPU_THROW_UNLESS(rules.size() == 1, "There cannot be more than one iterable rule with the same source and destination");
        return rules.front();
    };

    auto allTheSame = [](const std::vector<ie::DataPtr>& dataObjects) -> bool {
        if (dataObjects.empty()) {
            return true;
        }

        const auto& first = dataObjects.front();
        return std::all_of(dataObjects.begin(), dataObjects.end(),
            [&first](const ie::DataPtr& current) { return first->getTensorDesc() == current->getTensorDesc(); });
    };

    auto introduceLoopStart = [&]() -> Stage {
        // there may be several back-edge connections with the same pair of Tensor Iterator's input data object and body's output data object,
        // but different body's input data objects - they represent the same back-edge connection
        // nevertheless, we need to keep track of all body's input data object to correctly connect Loop Start's outputs and body's input stages
        std::map<std::pair<ie::DataPtr, ie::DataPtr>, std::vector<ie::DataPtr>> backedges;

        // iteration component inside Tensor Iterator's body can be used as an input for several stages at the same time
        std::map<std::pair<ie::DataPtr, IterationRule>, std::vector<ie::DataPtr>> iterations;
        std::map<ie::DataPtr, std::vector<ie::DataPtr>> intermediateDataObjects;

        // some Tensor Iterator's input data objects may be connected with several Tensor Iterator's body input data objects at the same time
        // back-edge connection is defined as a connection between Tensor Iterator's body output object and body input object
        // this way there can be different back-edge connections to the same Tensor Iterator's input object
        // to correctly handle this case we have to parse body inputs, not Tensor Iterator's inputs
        const auto& bodyInputs = tensorIterator->body.inputs;
        VPU_THROW_UNLESS(!bodyInputs.empty(), "If there is no an input for Tensor Iterator's body, so there is no iteration in tensor iterator");

        for (auto iterator = bodyInputs.begin(); iterator != bodyInputs.end(); ++iterator) {
            const auto& bodyInput = *iterator;
            const bool isLast = iterator == std::prev(bodyInputs.end());
            VPU_THROW_UNLESS(!isFakeHolder(bodyInput) || isLast , "There can be only one fake holder and it can be only the last Tensor Iterator body input");
            if (isFakeHolder(bodyInput)) {
                // fake holder keeps strong references on const data objects that are not presented in Tensor Iterator's body input vector
                // these const data objects will be process during parsing Tensor Iterator's body layers
                continue;
            }

            VPU_THROW_UNLESS(!(isIterable(bodyInput, tensorIterator) && hasBackEdgeConnectionTo(bodyInput, tensorIterator)),
                "There must not be a back-edge connection to iterable component");

            const auto& tensorIteratorInputs = findTIInputsDataByBodyData(bodyInput);
            VPU_THROW_UNLESS(tensorIteratorInputs.size() == 1,
                "There must be exactly one Tensor Iterator's input data object for each body's input data object except fake holder");
            const auto& tensorIteratorInput = tensorIteratorInputs.front();

            if (isIterable(bodyInput, tensorIterator)) {
                const auto& rule = getInputIterableRule(tensorIteratorInput, bodyInput);
                auto perm = DimsOrder::fromNumDims(tensorIteratorInput->getDims().size()).toPermutation();
                auto axis = perm[tensorIteratorInput->getDims().size() - 1 - rule.axis];
                iterations[std::make_pair(tensorIteratorInput, IterationRule{axis, rule.start, rule.stride, rule.end})].push_back(bodyInput);
            } else if (hasBackEdgeConnectionTo(bodyInput, tensorIterator)) {
                const auto& bodyOutputs = getBodyOutputsByBodyInput(bodyInput);
                VPU_THROW_UNLESS(bodyOutputs.size() == 1,
                    "There must be exactly one Tensor Iterator's body output data object for each back-edge connection "
                    "with the same Tensor Iterator's body input data object");
                const auto& bodyOutput = bodyOutputs.front();

                backedges[std::make_pair(bodyOutput, tensorIteratorInput)].push_back(bodyInput);
            } else {
                VPU_THROW_UNLESS(!isConst(bodyInput), "Const inputs of Tensor Iterator's body are hold by fake holder");
                VPU_THROW_UNLESS(!findTIInputsDataByBodyData(bodyInput).empty(), "There must be corresponding Tensor Iterator's input data object");

                intermediateDataObjects[tensorIteratorInput].push_back(bodyInput);
            }
        }

        auto loopStartInputs  = DataVector{};
        auto loopStartOutputs = DataVector{};

        for (const auto& backedge : backedges) {
            const auto& tensorIteratorInput = backedge.first.second;
            const auto& vpuTensorIteratorInput = getVpuData(tensorIteratorInput);
            VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

            auto loopStartInput = vpuTensorIteratorInput;
            if (!vpuTensorIteratorInput->canHaveAParent()) {
                auto copied = model->addNewData(vpuTensorIteratorInput->name() + "@copy-for-backedge", vpuTensorIteratorInput->desc());
                _stageBuilder->addCopyStage(model, "copy-for-backedge", nullptr, vpuTensorIteratorInput, copied, "copy for backedge");
                loopStartInput = copied;
            }

            const auto& backedgeInputs = backedge.second;
            VPU_THROW_UNLESS(allTheSame(backedgeInputs), "Different data objects cannot be mapped into the same data object");
            VPU_THROW_UNLESS(!backedgeInputs.empty(), "Back-edges are specified only from body output to body input");
            const auto& backedgeInput = backedgeInputs.front();

            VPU_THROW_UNLESS(getVpuData(backedgeInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");
            auto loopStartOutput = createData(backedgeInput);

            // to introduce shared data allocation edge later in Middle-End
            loopStartInput->attrs().set<Data>("start-shared-allocation", loopStartOutput);

            loopStartInputs.push_back(loopStartInput);
            loopStartOutputs.push_back(loopStartOutput);

            for (const auto& data : backedgeInputs) {
                bindData(loopStartOutput, data);
            }
        }

        IterationComponents start_iteration_components;
        for (const auto& iteration : iterations) {
            const auto& tensorIteratorInput = iteration.first.first;
            const auto& rule = iteration.first.second;
            const auto& vpuTensorIteratorInput = getVpuData(tensorIteratorInput);
            VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

            const auto& loopStartInput = vpuTensorIteratorInput;

            const auto& iterationInputs = iteration.second;
            VPU_THROW_UNLESS(allTheSame(iterationInputs), "Different data objects cannot be mapped into the same data object");
            VPU_THROW_UNLESS(!iterationInputs.empty(), "Iteration components are specified only from Tensor Iterator's input to body input");
            const auto& iterationInput = iterationInputs.front();

            VPU_THROW_UNLESS(getVpuData(iterationInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");
            auto loopStartOutput = createData(iterationInput);

            start_iteration_components.emplace(std::make_pair(loopStartInputs.size(), rule), loopStartOutputs.size());
            loopStartInputs.push_back(loopStartInput);
            loopStartOutputs.push_back(loopStartOutput);

            for (const auto& data : iterationInputs) {
                bindData(loopStartOutput, data);
            }
        }

        for (const auto& intermediateDataObject : intermediateDataObjects) {
            const auto& tensorIteratorInput = intermediateDataObject.first;
            const auto& vpuTensorIteratorInput = getVpuData(tensorIteratorInput);
            VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

            const auto& loopStartInput = vpuTensorIteratorInput;

            const auto& intermediateDataInputs = intermediateDataObject.second;
            VPU_THROW_UNLESS(allTheSame(intermediateDataInputs), "Different data objects cannot be mapped into the same data object");
            VPU_THROW_UNLESS(!intermediateDataInputs.empty(), "There must be at least one corresponding data object as body's input");

            const auto& intermediateDataInput = intermediateDataInputs.front();
            VPU_THROW_UNLESS(getVpuData(intermediateDataInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");

            const auto& loopStartOutput = createData(intermediateDataInput);
            bindData(loopStartOutput, intermediateDataInput);

            // to introduce shared data allocation edge later in Middle-End
            loopStartInput->attrs().set<Data>("start-shared-allocation", loopStartOutput);

            loopStartInputs.push_back(loopStartInput);
            loopStartOutputs.push_back(loopStartOutput);

            for (const auto& data : intermediateDataInputs) {
                bindData(loopStartOutput, data);
            }
        }

        auto loopStart = _stageBuilder->addLoopStartStage(model, tensorIterator->name + "@LoopStart", loopStartInputs, loopStartOutputs);
        loopStart->attrs().set("start-iteration-components", start_iteration_components);
        for (const auto& backedge : backedges) {
            const auto& parent = getVpuData(backedge.first.first);
            VPU_THROW_UNLESS(parent != nullptr, "Loop End's inputs must be already parsed");

            const auto& child = getVpuData(backedge.second.front());
            VPU_THROW_UNLESS(child != nullptr, "Loop Start's outputs must be already parsed");

            const auto& src_copy = parent;
            auto dst_copy = model->duplicateData(child, "@copy-for-backedge");
            for (const auto& consumerEdge : src_copy->consumerEdges()) {
                model->replaceStageInput(consumerEdge, dst_copy);
            }

            _stageBuilder->addCopyStage(model, "copy-for-backedge", nullptr, src_copy, dst_copy, "copy for backedge");

            // keep track of back-edges to introduce shared data allocation edges in Middle-End
            loopStart->attrs().getOrSet<HandleMultiMap<DataNode, Data>>("backedges", {}).emplace(dst_copy, child);
        }

        return loopStart;
    };

    auto introduceLoopEnd = [&]() -> Stage {
        std::map<std::pair<ie::DataPtr, IterationRule>, ie::DataPtr> iterations;
        std::map<ie::DataPtr, ie::DataPtr> intermediateDataObjects;

        auto loopEndInputs = DataVector{};

        const auto& bodyOutputs = tensorIterator->body.outputs;
        VPU_THROW_UNLESS(!bodyOutputs.empty(), "If there is no an output for Tensor Iterator's body, so there is no iteration in tensor iterator");

        for (const auto& bodyOutput : bodyOutputs) {
            VPU_THROW_UNLESS(!isFakeHolder(bodyOutput), "Fake holder can be only in body's input");

            const auto& tensorIteratorOutputs = findTIOutputsDataByBodyData(bodyOutput);
            VPU_THROW_UNLESS(tensorIteratorOutputs.empty() || tensorIteratorOutputs.size() == 1,
                "There may be only one Tensor Iterator's output data object for body's output data object if any");

            if (tensorIteratorOutputs.empty()) {
                // there can be no Tensor Iterator's output data object for body's output
                // in such a case there is no consumer for this data object, however, it's not a network's output
                // in this case we connect this data object with Loop End
                VPU_THROW_UNLESS(!isIterable(bodyOutput, tensorIterator),
                    "Body's output with no corresponding Tensor Iterator's output data object cannot be iterable component");

                auto loopEndInput = createData(bodyOutput);
                bindData(loopEndInput, bodyOutput);
                loopEndInputs.push_back(loopEndInput);
            } else {
                const auto& tensorIteratorOutput = tensorIteratorOutputs.front();
                if (isIterable(bodyOutput, tensorIterator)) {
                    const auto& rule = getOutputIterableRule(tensorIteratorOutput, bodyOutput);
                    auto perm = DimsOrder::fromNumDims(tensorIteratorOutput->getDims().size()).toPermutation();
                    auto axis = perm[tensorIteratorOutput->getDims().size() - 1 - rule.axis];
                    iterations[std::make_pair(tensorIteratorOutput, IterationRule{axis, rule.start, rule.stride, rule.end})] = bodyOutput;
                } else {
                    VPU_THROW_UNLESS(intermediateDataObjects.count(tensorIteratorOutput) == 0,
                        "There can be only one body's output data object for Tensor Iterator's output");
                    VPU_THROW_UNLESS(!isConst(tensorIteratorOutput), "Tensor Iterator's body cannot have const data object as output");

                    intermediateDataObjects[tensorIteratorOutput] = bodyOutput;
                }
            }
        }

        auto loopEndOutputs = DataVector{};

        IterationComponents end_iteration_components;
        for (const auto& iteration : iterations) {
            const auto& tensorIteratorOutput = iteration.first.first;
            const auto& rule = iteration.first.second;
            const auto& vpuTensorIteratorOutput = getVpuData(tensorIteratorOutput);
            VPU_THROW_UNLESS(vpuTensorIteratorOutput != nullptr, "Tensor Iterator's outputs must be parsed already");

            const auto& loopEndOutput = vpuTensorIteratorOutput;

            const auto& iterationInput = iteration.second;
            VPU_THROW_UNLESS(getVpuData(iterationInput) == nullptr, "Tensor Iterator's body output data objects were not parsed yet");
            auto loopEndInput = createData(iterationInput);

            end_iteration_components.emplace(std::make_pair(loopEndOutputs.size(), rule), loopEndInputs.size());
            loopEndInputs.push_back(loopEndInput);
            loopEndOutputs.push_back(loopEndOutput);

            bindData(loopEndInput, iterationInput);
        }

        for (const auto& intermediateDataObject : intermediateDataObjects) {
            const auto& tensorIteratorOutput = intermediateDataObject.first;
            const auto& vpuTensorIteratorOutput = getVpuData(tensorIteratorOutput);
            VPU_THROW_UNLESS(vpuTensorIteratorOutput != nullptr, "Tensor Iterator's outputs must be parsed already");

            auto loopEndOutput = vpuTensorIteratorOutput;
            if (loopEndOutput->usage() == DataUsage::Output) {
                auto to_copy = model->addNewData(loopEndOutput->name() + "@copy-for-backedge", loopEndOutput->desc());
                _stageBuilder->addCopyStage(model, "copy-for-tensor-iterator-output", nullptr, to_copy, loopEndOutput, "copy for TI output");
                loopEndOutput = to_copy;
            }

            const auto& intermediateDataInput = intermediateDataObject.second;
            VPU_THROW_UNLESS(getVpuData(intermediateDataInput) == nullptr, "Tensor Iterator's body output data objects were not parsed yet");

            auto loopEndInput = createData(intermediateDataInput);

            // to introduce shared data allocation edge later in Middle-End
            loopEndOutput->attrs().set<Data>("end-shared-allocation", loopEndInput);

            loopEndInputs.push_back(loopEndInput);
            loopEndOutputs.push_back(loopEndOutput);

            bindData(loopEndInput, intermediateDataInput);
        }

        auto loopEnd = _stageBuilder->addLoopEndStage(model, tensorIterator->name + "@LoopEnd", loopEndInputs, loopEndOutputs);
        loopEnd->attrs().set("end-iteration-components", end_iteration_components);
        return loopEnd;
    };

    // Loop End must be introduced first to parse Tensor Iterator's body output data objects before parsing back-edge connections
    auto loopEnd = introduceLoopEnd();
    auto loopStart = introduceLoopStart();

    loopStart->attrs().set<uint32_t>("iterations-count", getNumIteration(*tensorIterator));
    loopEnd->attrs().set<uint32_t>("iterations-count", getNumIteration(*tensorIterator));

    // to allocate LoopEnd and LoopStart at the same time
    loopStart->attrs().set<Stage>("loop-end", loopEnd);

    // to be sure all loop's inputs are still alive during loop execution
    // force they to be alive as long as loop's outputs
    for (const auto& loopStartInput : loopStart->inputs()) {
        model->addStageInput(loopEnd, loopStartInput);
    }

    for (const auto& bodyLayer : ie::NetPass::TIBodySortTopologically(tensorIterator->body)) {
        if (bodyLayer->type == "Const") {
            // since Tensor Iterator's body is a kind of CNNNetwork it may has const data objects as inputs
            // const data objects are hold by "Const" layers
            // we don't need them during iteration and ignore the same way as compilation process of regular network
            continue;
        }

        auto stageInputs = DataVector{};
        for (const auto& data : bodyLayer->insData) {
            const auto ieInput = data.lock();
            const auto vpuInput = isConst(ieInput) ? createConstData(ieInput) : getVpuData(ieInput);
            VPU_THROW_UNLESS(vpuInput != nullptr,
                "Non-const input of a stage must be already parsed due to either topological order or as Loop Start's output");

            stageInputs.push_back(vpuInput);
        }

        auto stageOutputs = DataVector{};
        for (const auto& data : bodyLayer->outData) {
            auto output = getVpuData(data);
            // output of a stage might be already parsed as Loop End's output
            if (output == nullptr) {
                output = createData(data);
                bindData(output, data);
            }

            stageOutputs.push_back(output);
        }

        parseLayer(model, bodyLayer, stageInputs, stageOutputs);
    }
}

}  // namespace vpu
