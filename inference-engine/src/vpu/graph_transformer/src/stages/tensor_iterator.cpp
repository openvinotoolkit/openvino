// // Copyright (C) 2018-2021 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include "vpu/frontend/frontend.hpp"
// #include "vpu/stages/iteration_rule.hpp"
// #include "vpu/utils/auto_scope.hpp"
// #include <legacy/graph_transformer.h>
// #include "vpu/model/data_contents/ie_blob_content.hpp"

// #include <legacy/ie_layers_internal.hpp>
// #include <legacy/net_pass.h>

// #include <memory>
// #include <utility>
// #include <vector>
// #include <map>
// #include <string>

// namespace vpu {

// namespace {
// struct PortMap {
//     // Data map rule
//     int from; /**< Index of exteral data from ins/outs fields of CNNLayer */
//     int to;   /**< Index of internal data in iterator body */

//     // Iteration rule
//     int axis;      /**< Axis to iterate throught */
//     int stride;    /**< Stride to iterate throught */
//     int start;     /**< Start index of iteration range */
//     int end;       /**< Last index of iteration range  */
//     int part_size; /**< Part size which will be transfered to body subnetwork */
// };

// constexpr auto s_curIterPort   = "loop_body_current_iteration_idx";
// constexpr auto s_tripCountPort = "loop_trip_count_idx";
// constexpr auto s_initCondPort  = "loop_execution_condition_idx";
// constexpr auto s_condPort      = "loop_body_condition_output_idx";

// bool isIterable(const PortMap& rule) {
//     return rule.axis != -1;
// }

// bool isIterableInput(const size_t& bodyInputIdx, const std::shared_ptr<ngraph::opset4::TensorIterator>& tensorIterator) {
//     auto descriptions = tensorIterator->get_input_descriptions();
    
//     auto desc = *std::find_if(descriptions.begin(), descriptions.end(), [&bodyInputIdx](ngraph::op::util::SubGraphOp::InputDescription& desc) {
//         return desc.m_body_parameter_index == bodyInputIdx;
//     });

//     if (auto sliceInputDesc = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc)) {
//         return sliceInputDesc->m_axis != -1;
//     }
//     return false;


//     // const auto isInput = [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->get_body()->get_parameters(rule.to) .inputs[rule.to] == data; };
//     // const auto& rules = tensorIterator->input_port_map;
//     // return std::any_of(rules.begin(), rules.end(), [&isInput](const PortMap& rule) { return isIterable(rule) && isInput(rule); });
// }

// bool isIterableOutput(const size_t& bodyResultIdx , const std::shared_ptr<ngraph::opset4::TensorIterator>& tensorIterator) {


//     auto descriptions = tensorIterator->get_output_descriptions();
//     auto desc = *std::find_if(descriptions.begin(), descriptions.end(), [&bodyResultIdx](ngraph::op::util::SubGraphOp::OutputDescription& desc) {
//         return desc.m_body_value_index == bodyResultIdx;
//     });
//     if (auto concatOutputDesc = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(desc)) {
//         return concatOutputDesc->m_axis != -1;
//     }
//     return false;


//     // const auto isOutput = [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->body.outputs[rule.to] == data; };
//     // const auto& rules = tensorIterator->output_port_map;
//     // return std::any_of(rules.begin(), rules.end(), [&isOutput](const PortMap& rule) { return isIterable(rule) && isOutput(rule); });
// }

// // bool isIterable(const ie::DataPtr& data, const std::shared_ptr<ngraph::opset4::TensorIterator>& tensorIterator) {
// //     const auto& bodyInputs = tensorIterator->body.inputs;
// //     const auto& bodyOutputs = tensorIterator->body.outputs;

// //     const bool isBodyInput = std::find(bodyInputs.begin(), bodyInputs.end(), data) != bodyInputs.end();
// //     const bool isBodyOutput = std::find(bodyOutputs.begin(), bodyOutputs.end(), data) != bodyOutputs.end();
// //     VPU_THROW_UNLESS(isBodyInput || isBodyOutput, "Check on iterable component is valid only for Tensor Iterator's body input and output data objects");

// //     return isIterableInput(data, tensorIterator) || isIterableOutput(data, tensorIterator);
// // }



// bool isConst(const ie::CNNLayerPtr& layer) {
//     return layer->type == "Const" && layer->outData.size() == 1 && layer->blobs.size() == 1;
// }

// bool isConst(const NodePtr& data) {
//     const auto creator = data->get_input_node_shared_ptr(0);
//     return ngraph::as_type_ptr<ngraph::opset4::Constant>(data) != nullptr && creator->get_output_size() == 1;
// }

// bool isFakeHolder(const NodePtr& data) {
//     return data->get_output_tensor(0).get_element_type() == ngraph::element::undefined;
// }

// // std::vector<PortMap> getInputPortMap (const std::vector<ngraph::op::util::InputDescriptionPtr>& descriptions) {
// //     std::vector<PortMap> result;
// //     result.reserve(descriptions.size());
// //     for (const auto& desc : descriptions) {
// //         auto body_input_index = desc->m_body_parameter_index;

// //         if (const auto slice_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc)) {
// //             result.push_back(PortMap{
// //                 static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
// //                 static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
// //                 static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
// //                 static_cast<int>(slice_desc->m_part_size)});
// //         } else if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc)) {
// //             result.push_back(PortMap {
// //                 static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
// //         } else if (const auto inv_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
// //             result.push_back(PortMap {
// //                     static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
// //         } else {
// //             VPU_THROW_UNLESS(false, "Incorrect type of the input description.");
// //         }
// //     }
// //     return result;
// // }

// // std::vector<PortMap> getOutputPortMap (const std::vector<ngraph::op::util::OutputDescriptionPtr>& descriptions) {
// //     std::vector<PortMap> result;
// //     result.reserve(descriptions.size());
// //         for (const auto& desc : descriptions) {
// //         auto body_output_idx = desc->m_body_value_index;

// //         std::string type_name = desc->get_type_info().name;
// //         if (type_name == "ConcatOutputDescription") {
// //             auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(desc);
// //             IE_ASSERT(output_desc != nullptr);

// //             result.push_back(PortMap {
// //                 static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
// //                 static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
// //                 static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
// //                 static_cast<int>(output_desc->m_part_size)});
// //         } else if (type_name == "BodyOutputDescription") {
// //             auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::BodyOutputDescription>(desc);
// //             IE_ASSERT(output_desc != nullptr);

// //             result.push_back(PortMap {
// //                 static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
// //         } else {
// //             VPU_THROW_UNLESS(false, "Incorrect type of the input description.");
// //         }
// //     }
// //     return result;
// // }

// std::vector<PortMap> getBackEdges (const std::vector<ngraph::op::util::InputDescriptionPtr>& descriptions) {
//     std::vector<PortMap> result;
//     result.reserve(descriptions.size());
//     for (const auto& desc : descriptions) {
//         auto body_input_index = desc->m_body_parameter_index;
//         if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc)) {
//             auto body_output_idx = merge_desc->m_body_value_index;

//             result.push_back(PortMap {
//                 static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
//         }
//     }
//     return result;
// }

// }  // namespace

// void FrontEnd::parseTensorIterator(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) {
//     IE_ASSERT(!inputs.empty());
//     IE_ASSERT(!outputs.empty());
//     // auto tensorIterator = std::dynamic_pointer_cast<ie::TensorIterator>(ie::CNNLayerPtr());
//     // IE_ASSERT(tensorIterator != nullptr);


//     auto tensorIterator = ngraph::as_type_ptr<ngraph::opset4::TensorIterator>(node);
//     auto subgraph = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp>(node);
//     auto body = tensorIterator->get_body();
//     auto bodyParams = body->get_parameters();
//     auto bodyResults = body->get_results();
//     auto bodyAsCNNNetwork = ie::CNNNetwork(body);
//     auto bodyInputsInfo = bodyAsCNNNetwork.getInputsInfo();
//     auto bodyOutputsInfo = bodyAsCNNNetwork.getOutputsInfo();
//     auto bodyInputDataVector = [&bodyInputsInfo] {
//          std::vector<ie::DataPtr> result;
//          result.reserve(bodyInputsInfo.size());
//          for (const auto& inputInfo : bodyInputsInfo) {
//              result.push_back(inputInfo.second->getInputData());
//          }
//          return result;
//     }();
//     auto bodyOutputDataVector = [&bodyOutputsInfo] {
//          std::vector<ie::DataPtr> result;
//          result.reserve(bodyOutputsInfo.size());
//          for (const auto& inputInfo : bodyOutputsInfo) {
//              result.push_back(inputInfo.second);
//          }
//          return result;
//     }();
//     auto inputDescriptions = subgraph->get_input_descriptions();
//     auto outputDescriptions = subgraph->get_output_descriptions();

//     IE_ASSERT(bodyParams.size() == bodyInputsInfo.size());
//     IE_ASSERT(bodyResults.size() == bodyResults.size());

//     auto createDescriptor = [&](const ngraph::descriptor::Tensor& original) {
//         auto vpuDescriptor = DataDesc{original};
//         if (vpuDescriptor.type() == DataType::FP32) {
//             // to infer the same FP32 models on different devices (CPU, GPU, VPU and so on)
//             vpuDescriptor.setType(DataType::FP16);
//         }
//         return vpuDescriptor;
//     };

//     auto createData = [&](const NodePtr& original) {
        
//         return model->addNewData(original->get_friendly_name(), createDescriptor(original->get_input_tensor(0)));;
//     };

//     auto createConstData = [&](const NodePtr& original) {
//         VPU_THROW_UNLESS(isConst(original), "VPU const data object can be created only from const IE data object");
//         // auto constantNode = ngraph::as_type_ptr<ngraph::opset4::Constant>(original);
//         // const auto& creator = getCreatorLayer(original).lock();
//         // const auto& blob = ieBlobContent(creator->blobs.begin()->second, descriptor.type());
//         const auto origWeights = shareWeights_(original);
        
//         const auto& descriptor = DataDesc({origWeights->size()});

//         return model->addConstData(original->get_friendly_name(), descriptor, ieBlobContent(origWeights));
//     };

//     auto findInputDescriptionByBodyInputIdx = [&](const size_t& bodyInputIdx) {
//         auto descriptions = tensorIterator->get_input_descriptions();
    
//         auto desc = *std::find_if(descriptions.begin(), descriptions.end(), [&bodyInputIdx](ngraph::op::util::SubGraphOp::InputDescription& desc) {
//             return desc.m_body_parameter_index == bodyInputIdx;
//         });
//         return desc;
//     };
//     auto findTIInputNodeByBodyInputIdx = [&](const size_t& bodyInputIdx) {
//         // std::vector<ie::DataPtr> tensorIteratorInputs;
//         auto desc = findInputDescriptionByBodyInputIdx(bodyInputIdx);
//         return subgraph->input(desc->m_input_index);
//         // auto inputPortMap = getInputPortMap(tensorIterator->get_input_descriptions());
//         // for (const auto& rule : inputPortMap) {
//         //     if (tensorIterator->body.inputs[rule.to] == bodyData) {
//         //         tensorIteratorInputs.push_back(tensorIterator->insData[rule.from].lock());
//         //     }
//         // }
//         // return tensorIteratorInputs;
//     };
//     auto hasBackEdgeConnectionTo = [&](const size_t& bodyInputIdx) {
//         auto desc = findInputDescriptionByBodyInputIdx(bodyInputIdx);
//         if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc)) {
//             return true;
//         }
//         return false;
//         // const auto& rules = tensorIterator->back_edges;
//         // return std::any_of(rules.begin(), rules.end(), [&data, &tensorIterator](const PortMap& rule) { return tensorIterator->body.inputs[rule.to] == data; });
//     };
//     auto findOutputDescriptionByBodyOutputIdx = [&](const size_t& bodyOutputIdx) {
//         auto descriptions = tensorIterator->get_output_descriptions();
    
//         auto desc = *std::find_if(descriptions.begin(), descriptions.end(), [&bodyOutputIdx](ngraph::op::util::SubGraphOp::OutputDescription& desc) {
//             return desc.m_body_value_index == bodyOutputIdx;
//         });
//         return desc;
//     };
//     auto findTIOutputsDataByBodyResultIdx = [&](const size_t& bodyResultsIdx) -> ngraph::Output<ngraph::Node> {
//         auto desc = findOutputDescriptionByBodyOutputIdx(bodyResultsIdx);
//             VPU_THROW_UNLESS(subgraph->get_output_size() >= bodyResultsIdx,
//                 "Can't get TI output by param idx: subgraph output size = {} but provided idx = {}", subgraph->get_output_size(), bodyResultsIdx);
//         return subgraph->output(bodyResultsIdx);


//         // for (const auto& rule : tensorIterator->output_port_map) {
//         //     if (tensorIterator->body.outputs[rule.to] == bodyData) {
//         //         tensorIteratorOutputs.push_back(tensorIterator->outData[rule.from]);
//         //     }
//         // }
//         // return tensorIteratorOutputs;
//     };

//     auto getBodyOutputsByBodyInput = [&](const size_t& bodyInputIdx) {
//         std::vector<ie::DataPtr> bodyOutputs;
//         // auto desc = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::> findInputDescriptionByBodyInputIdx(bodyInputIdx);
//         auto desc = findInputDescriptionByBodyInputIdx(bodyInputIdx);
//         const auto mergeDesc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc);
//         return bodyResults[mergeDesc->m_body_value_index];


//         // auto body_output_idx = merge_desc->m_body_value_index;
//         // for (const auto& rule : tensorIterator->back_edges) {
//         //     if (tensorIterator->body.inputs[rule.to] == bodyInput) {
//         //         bodyOutputs.push_back(tensorIterator->body.outputs[rule.from]);
//         //     }
//         // }
//         // return bodyOutputs;
//     };

//     // auto getInputIterableRule = [&](const ie::DataPtr& from, const ie::DataPtr& to) {
//     //     std::vector<PortMap> rules;
//     //     for (const auto& rule : tensorIterator->input_port_map) {
//     //         if (isIterable(rule) && tensorIterator->insData[rule.from].lock() == from && tensorIterator->body.inputs[rule.to] == to) {
//     //             rules.push_back(rule);
//     //         }
//     //     }
//     //     VPU_THROW_UNLESS(!rules.empty(), "There must be an iterable rule between data objects");
//     //     VPU_THROW_UNLESS(rules.size() == 1, "There cannot be more than one iterable rule with the same source and destination");
//     //     return rules.front();
//     // };

//     // auto getOutputIterableRule = [&](const ie::DataPtr& from, const ie::DataPtr& to) {
//     //     std::vector<PortMap> rules;
//     //     for (const auto& rule : tensorIterator->output_port_map) {
//     //         if (isIterable(rule) && tensorIterator->outData[rule.from] == from && tensorIterator->body.outputs[rule.to] == to) {
//     //             rules.push_back(rule);
//     //         }
//     //     }
//     //     VPU_THROW_UNLESS(!rules.empty(), "There must be an iterable rule between data objects");
//     //     VPU_THROW_UNLESS(rules.size() == 1, "There cannot be more than one iterable rule with the same source and destination");
//     //     return rules.front();
//     // };

//     auto allTheSame = [](const std::vector<ie::DataPtr>& dataObjects) -> bool {
//         if (dataObjects.empty()) {
//             return true;
//         }

//         const auto& first = dataObjects.front();
//         return std::all_of(dataObjects.begin(), dataObjects.end(),
//             [&first](const ie::DataPtr& current) { return first->getTensorDesc() == current->getTensorDesc(); });
//     };

//     auto introduceLoopStart = [&]() -> Stage {
//         // there may be several back-edge connections with the same pair of Tensor Iterator's input data object and body's output data object,
//         // but different body's input data objects - they represent the same back-edge connection
//         // nevertheless, we need to keep track of all body's input data object to correctly connect Loop Start's outputs and body's input stages
//         std::map<std::pair<std::shared_ptr<ngraph::op::v0::Result>, const ngraph::Input<ngraph::Node>>, std::shared_ptr<ngraph::op::v0::Parameter>> backedges;

//         // iteration component inside Tensor Iterator's body can be used as an input for several stages at the same time
//         std::map<std::pair<const ngraph::Input<ngraph::Node>, IterationRule>, std::shared_ptr<ngraph::op::v0::Parameter>> iterations;
//         std::map<const ngraph::Input<ngraph::Node>, std::shared_ptr<ngraph::op::v0::Parameter>> intermediateDataObjects;

//         // some Tensor Iterator's input data objects may be connected with several Tensor Iterator's body input data objects at the same time
//         // back-edge connection is defined as a connection between Tensor Iterator's body output object and body input object
//         // this way there can be different back-edge connections to the same Tensor Iterator's input object
//         // to correctly handle this case we have to parse body inputs, not Tensor Iterator's inputs
//         // const auto& bodyInputs = tensorIterator->body.inputs;
//         VPU_THROW_UNLESS(!bodyInputsInfo.empty(), "If there is no an input for Tensor Iterator's body, so there is no iteration in tensor iterator");
//         std::size_t bodyInputIdx = 0;
//         // for (std::size_t bodyInputPort = 0; bodyInputPort < bodyInputs.size(); ++bodyInputPort) {
//         for (auto bodyParam : bodyParams) {    
//             // const auto& bodyInput = bodyInputInfo.second->getInputData();
//             const bool isLast = bodyInputIdx == (bodyParams.size() - 1);
//             bodyInputIdx++;
//             // VPU_THROW_UNLESS(!isFakeHolder(bodyInput) || isLast , "There can be only one fake holder and it can be only the last Tensor Iterator body input");
//             // if (isFakeHolder(bodyInput)) {
//             //     // fake holder keeps strong references on const data objects that are not presented in Tensor Iterator's body input vector
//             //     // these const data objects will be process during parsing Tensor Iterator's body layers
//             //     continue;
//             // }

//             // VPU_THROW_UNLESS(!(isIterableInput(bodyInputIdx, tensorIterator) && hasBackEdgeConnectionTo(bodyInput, tensorIterator)),
//             //                  "There must not be a back-edge connection to iterable component");

//             const auto& tensorIteratorInput = findTIInputNodeByBodyInputIdx(bodyInputIdx);
//             // VPU_THROW_UNLESS(tensorIteratorInputs.size() == 1,
//             //                  "There must be exactly one Tensor Iterator's input data object for each body's input data object except fake holder");
//             // const auto& tensorIteratorInput = tensorIteratorInputs.front();

//             if (isIterableInput(bodyInputIdx, tensorIterator)) {
//                 // const auto& rule = getInputIterableRule(tensorIteratorInput, bodyInput);

//                 auto perm = DimsOrder::fromNumDims(tensorIteratorInput.get_shape().size()).toPermutation();
//                 auto desc = findInputDescriptionByBodyInputIdx(bodyInputIdx);
//                 const auto sliceDesc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc);
//                 auto axis = perm[tensorIteratorInput.get_shape().size() - 1 - sliceDesc->m_axis];
//                 iterations[std::make_pair(tensorIteratorInput, IterationRule{axis, sliceDesc->m_start, sliceDesc->m_stride, sliceDesc->m_end})] = (bodyParam);
//             } else if (hasBackEdgeConnectionTo(bodyInputIdx)) {
//                 const auto& bodyOutput = getBodyOutputsByBodyInput(bodyInputIdx);
//                 // VPU_THROW_UNLESS(bodyOutputs.size() == 1,
//                 //                  "There must be exactly one Tensor Iterator's body output data object for each back-edge connection "
//                 //                  "with the same Tensor Iterator's body input data object");
//                 // const auto& bodyOutput = bodyOutputs.front();

//                 backedges[std::make_pair(bodyOutput, tensorIteratorInput)] = (bodyParams[bodyInputIdx]);
//             } else {
//                 // VPU_THROW_UNLESS(!isConst(bodyInput), "Const inputs of Tensor Iterator's body are hold by fake holder");
//                 // VPU_THROW_UNLESS(!findTIInputsDataByBodyData(bodyInput).empty(), "There must be corresponding Tensor Iterator's input data object");

//                 intermediateDataObjects[tensorIteratorInput] = (bodyParams[bodyInputIdx]);
//             }
//         }

//         auto loopStartInputs  = DataVector{};
//         auto loopStartOutputs = DataVector{};

//         for (const auto& backedge : backedges) {
//             const auto& tensorIteratorInput = backedge.first.second;
//             const auto& vpuTensorIteratorInput = getVpuData(std::shared_ptr<ngraph::Node>(tensorIteratorInput.get_node()));
//             VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

//             auto loopStartInput = vpuTensorIteratorInput;
//             if (!vpuTensorIteratorInput->canHaveAParent()) {
//                 auto copied = model->addNewData(vpuTensorIteratorInput->name() + "@copy-for-backedge", vpuTensorIteratorInput->desc());
//                 _stageBuilder->addCopyStage(model, "copy-for-backedge", nullptr, vpuTensorIteratorInput, copied, "copy for backedge");
//                 loopStartInput = copied;
//             }

//             const auto& backedgeInputs = backedge.second;
//             // VPU_THROW_UNLESS(allTheSame(backedgeInputs), "Different data objects cannot be mapped into the same data object");
//             // VPU_THROW_UNLESS(!backedgeInputs.empty(), "Back-edges are specified only from body output to body input");
//             const auto& backedgeInput = backedgeInputs;//.front();

//             VPU_THROW_UNLESS(getVpuData(backedgeInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");
//             auto loopStartOutput = createData(backedgeInput);

//             // to introduce shared data allocation edge later in Middle-End
//             loopStartInput->attrs().set<Data>("start-shared-allocation", loopStartOutput);

//             loopStartInputs.push_back(loopStartInput);
//             loopStartOutputs.push_back(loopStartOutput);

//             // for (const auto& data : backedgeInputs) {
//             //     bindData(loopStartOutput, data);
//             // }
//             // bindData(loopStartOutput, backedgeInput);
//         }

//         vpu::Optional<std::uint32_t> batchIdx{};
//         if (auto loop = ngraph::as_type_ptr<ngraph::opset6::Loop>(node)) {
//             auto specPort = loop->get_special_body_ports();// findInputDescriptionByBodyInputIdx(bodyInputIdx);
//             // if (tensorIterator->params.count(s_tripCountPort)) {
//             //     VPU_THROW_UNLESS(!iterations.empty(),
//             //         "Encountered Loop which is supposed to be loop by dynamic batch (dynamic iterations count), but didn't find an iteration component");
//             //     VPU_THROW_UNLESS(!tensorIterator->params.count(s_curIterPort), "Current iteration port for body of Loop operation is not supported");
//             //     batchIdx = static_cast<std::uint32_t>(loopStartInputs.size());
//             // }
//             // if (1) {
//             //     const auto& input = node->input(1);
//             //     // VPU_THROW_UNLESS(isConst(input), "Execution condition for Loop must be constant true");

//             //     const auto& constantNode = ngraph::as_type_ptr<ngraph::opset4::Constant>(std::shared_ptr<ngraph::Node>(input.get_node()));
//             //     // VPU_THROW_UNLESS(creator->blobs.size() == 1, "Execution condition for Loop must contain exactly one blob, got {}", creator->blobs.size());
//             //     VPU_THROW_UNLESS(constantNode != nullptr, "Execution condition for Loop must contain exactly one blob, got {}");
//             //     auto& inputTensor = input.get_tensor();
//             //     ie::Blob::Ptr blob = ie::Blob::Ptr({inputTensor});
//             //     const auto& blob = creator->blobs.begin()->second;
//             //     VPU_THROW_UNLESS(blob->size() == 1, "Execution condition for Loop must be single value, got {} values", blob->size());
//             //     VPU_THROW_UNLESS(blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32,
//             //                     "Execution condition for Loop must have I32 type, got {}", blob->getTensorDesc().getPrecision());

//             //     const auto value = blob->buffer().as<std::int32_t*>()[0];
//             //     VPU_THROW_UNLESS(value == 1, "Execution condition for Loop must be true, got {} as value", value);
//             // }
//         }
//             IterationComponents start_iteration_components;
//             for (const auto& iteration : iterations) {
//                 const auto& tensorIteratorInput = iteration.first.first;
//                 const auto& rule = iteration.first.second;
//                 const auto& vpuTensorIteratorInput = getVpuData(std::shared_ptr<ngraph::Node>(tensorIteratorInput.get_node()));
//                 VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

//                 const auto& loopStartInput = vpuTensorIteratorInput;

//                 const auto& iterationInputs = iteration.second;
//                 // VPU_THROW_UNLESS(allTheSame(iterationInputs), "Different data objects cannot be mapped into the same data object");
//                 // VPU_THROW_UNLESS(!iterationInputs.empty(), "Iteration components are specified only from Tensor Iterator's input to body input");
//                 const auto& iterationInput = iterationInputs->get_input_node_shared_ptr(0);

//                 VPU_THROW_UNLESS(getVpuData(iterationInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");
//                 auto loopStartOutput = createData(iterationInput);

//                 start_iteration_components.emplace(std::make_pair(loopStartInputs.size(), rule), loopStartOutputs.size());
//                 loopStartInputs.push_back(loopStartInput);
//                 loopStartOutputs.push_back(loopStartOutput);
//                 // bindData(loopStartOutput, iterationInput);
//                 // for (const auto& data : iterationInputs) {
//                 //     bindData(loopStartOutput, data);
//                 // }
//             }
        
//         for (const auto& intermediateDataObject : intermediateDataObjects) {
//             const auto& tensorIteratorInput = intermediateDataObject.first;
//             const auto& vpuTensorIteratorInput = getVpuData(std::shared_ptr<ngraph::Node>(tensorIteratorInput.get_node()));
//             VPU_THROW_UNLESS(vpuTensorIteratorInput != nullptr, "Tensor Iterator's inputs must be parsed already");

//             const auto& loopStartInput = vpuTensorIteratorInput;

//             const auto& intermediateDataInputs = intermediateDataObject.second;
//             // VPU_THROW_UNLESS(allTheSame(intermediateDataInputs), "Different data objects cannot be mapped into the same data object");
//             // VPU_THROW_UNLESS(!intermediateDataInputs.empty(), "There must be at least one corresponding data object as body's input");

//             const auto& intermediateDataInput = intermediateDataInputs;
//             VPU_THROW_UNLESS(getVpuData(intermediateDataInput) == nullptr, "Tensor Iterator's body input data objects were not parsed yet");

//             const auto& loopStartOutput = createData(intermediateDataInput);
//             // bindData(loopStartOutput, intermediateDataInput);

//             // to introduce shared data allocation edge later in Middle-End
//             loopStartInput->attrs().set<Data>("start-shared-allocation", loopStartOutput);

//             loopStartInputs.push_back(loopStartInput);
//             loopStartOutputs.push_back(loopStartOutput);
//             // bindData(loopStartOutput, intermediateDataInput);
//             // for (const auto& data : intermediateDataInputs) {
//             //     bindData(loopStartOutput, data);
//             // }
//         }

//         auto loopStart = _stageBuilder->addLoopStartStage(model, tensorIterator->get_friendly_name() + "@LoopStart", loopStartInputs, loopStartOutputs);
//         loopStart->attrs().set("start-iteration-components", start_iteration_components);
//         if (batchIdx.hasValue()) {
//             loopStart->attrs().set("batchId", batchIdx.get());
//         }

//         for (const auto& backedge : backedges) {
//             const auto& parent = getVpuData(backedge.first.first);
//             VPU_THROW_UNLESS(parent != nullptr, "Loop End's inputs must be already parsed");

//             const auto& child = getVpuData(backedge.second);
//             VPU_THROW_UNLESS(child != nullptr, "Loop Start's outputs must be already parsed");

//             const auto& src_copy = parent;
//             auto dst_copy = model->duplicateData(child, "@copy-for-backedge");
//             for (const auto& consumerEdge : src_copy->consumerEdges()) {
//                 model->replaceStageInput(consumerEdge, dst_copy);
//             }

//             _stageBuilder->addCopyStage(model, "copy-for-backedge", nullptr, src_copy, dst_copy, "copy for backedge");

//             // keep track of back-edges to introduce shared data allocation edges in Middle-End
//             loopStart->attrs().getOrSet<HandleMultiMap<DataNode, Data>>("backedges", {}).emplace(dst_copy, child);
//         }

//         return loopStart;
//     };

//     auto introduceLoopEnd = [&]() -> Stage {
//         std::map<std::pair<const ngraph::Output<ngraph::Node>, IterationRule>, std::shared_ptr<ngraph::op::v0::Result>> iterations;
//         std::map<const ngraph::Output<ngraph::Node>, std::shared_ptr<ngraph::op::v0::Result>> intermediateDataObjects;
//         // std::map<std::pair<ie::DataPtr, IterationRule>, ie::DataPtr> iterations;
//         // std::map<ie::DataPtr, ie::DataPtr> intermediateDataObjects;

//         auto loopEndInputs = DataVector{};

//         // const auto& bodyOutputs = tensorIterator->body.outputs;
//         VPU_THROW_UNLESS(!bodyResults.empty(), "If there is no an output for Tensor Iterator's body, so there is no iteration in tensor iterator");

//         for (std::size_t bodyResultsIdx = 0; bodyResultsIdx < bodyResults.size(); ++bodyResultsIdx) {
//             const auto& bodyOutput = bodyResults[bodyResultsIdx];

//             if (auto loop = ngraph::as_type_ptr<ngraph::opset6::Loop>(node)) {  // need to rework logic.
//                 auto specPort = loop->get_special_body_ports();
//                 auto curIterPort = specPort.current_iteration_input_idx;
//             // tensorIterator->params.count(s_condPort) && tensorIterator->GetParamAsUInt(s_condPort) == bodyOutputIdx) {
//                 if (auto condPort = specPort.body_condition_output_idx == bodyResultsIdx) {
//                     const auto& creator = bodyOutput->get_input_node_shared_ptr(0);
//                     if (!creator) {
//                         // ConstTransformer leaves constant without creator
//                         // Assume it's true
//                         continue;
//                     }
//                     VPU_THROW_UNLESS(isConst(bodyOutput), "Body execution condition must be constant true");

//                     // VPU_THROW_UNLESS(creator->blobs.size() == 1, "Body execution condition constant must have one blob");
//                     // const auto& blob = creator->blobs.begin()->second;
//                     // VPU_THROW_UNLESS(blob->size() == 1, "Body execution condition must be single value");
//                     // VPU_THROW_UNLESS(blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32, "Body execution condition must be I32");
//                     // const auto value = blob->buffer().as<std::int32_t*>()[0];
//                     // VPU_THROW_UNLESS(value == 1, "Body execution condition must be true");
//                     continue;
//                 }
//             }

//             VPU_THROW_UNLESS(!isFakeHolder(bodyOutput), "Fake holder can be only in body's input");

//             const auto& tensorIteratorOutput = findTIOutputsDataByBodyResultIdx(bodyResultsIdx);
//             // VPU_THROW_UNLESS(tensorIteratorOutputs.empty() || tensorIteratorOutputs.size() == 1,
//             //     "There may be only one Tensor Iterator's output data object for body's output data object if any");
            

//             // look once again!!!!!!
//             // if (tensorIteratorOutputs.empty()) {
//             //     // there can be no Tensor Iterator's output data object for body's output
//             //     // in such a case there is no consumer for this data object, however, it's not a network's output
//             //     // in this case we connect this data object with Loop End
//             //     VPU_THROW_UNLESS(!isIterable(bodyOutput, tensorIterator),
//             //         "Body's output with no corresponding Tensor Iterator's output data object cannot be iterable component");

//             //     auto loopEndInput = createData(bodyOutput);
//             //     bindData(loopEndInput, bodyOutput);
//             //     loopEndInputs.push_back(loopEndInput);
//             // } else 



//             {
//                 // const auto& tensorIteratorOutput = tensorIteratorOutputs.front();
//                 if (isIterableOutput(bodyResultsIdx, tensorIterator)) {
//                     // const auto& rule = getOutputIterableRule(tensorIteratorOutput, bodyOutput);
//                     // auto perm = DimsOrder::fromNumDims(tensorIteratorOutput->getDims().size()).toPermutation();
//                     // auto axis = perm[tensorIteratorOutput->getDims().size() - 1 - rule.axis];
//                     // iterations[std::make_pair(tensorIteratorOutput, IterationRule{axis, rule.start, rule.stride, rule.end})] = bodyOutput;


//                     auto perm = DimsOrder::fromNumDims(tensorIteratorOutput.get_shape().size()).toPermutation();
//                     auto desc = findOutputDescriptionByBodyOutputIdx(bodyResultsIdx);
//                     const auto sliceDesc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc);
//                     auto axis = perm[tensorIteratorOutput.get_shape().size() - 1 - sliceDesc->m_axis];
//                     iterations[std::make_pair(tensorIteratorOutput, IterationRule{axis, sliceDesc->m_start, sliceDesc->m_stride, sliceDesc->m_end})] = (bodyOutput);


//                 } else {
//                     VPU_THROW_UNLESS(intermediateDataObjects.count(tensorIteratorOutput) == 0,
//                         "There can be only one body's output data object for Tensor Iterator's output");
//                     VPU_THROW_UNLESS(!isConst(tensorIteratorOutput.get_node_shared_ptr()), "Tensor Iterator's body cannot have const data object as output");

//                     intermediateDataObjects[tensorIteratorOutput] = bodyOutput;
//                 }
//             }
//         }

//         auto loopEndOutputs = DataVector{};

//         vpu::Optional<std::uint32_t> batchIdx{};
//         // if (tensorIterator->params.count(s_tripCountPort)) {
//         //     VPU_THROW_UNLESS(!iterations.empty(),
//         //         "Encountered Loop which is supposed to be loop by dynamic batch (dynamic iterations count), but didn't find an iteration component");
//         //     VPU_THROW_UNLESS(!tensorIterator->params.count(s_curIterPort), "Current iteration port for body of Loop operation is not supported");
//         //     batchIdx = static_cast<std::uint32_t>(loopEndOutputs.size());
//         // }

//         IterationComponents end_iteration_components;
//         for (const auto& iteration : iterations) {
//             const auto& tensorIteratorOutput = iteration.first.first;
//             const auto& rule = iteration.first.second;
//             const auto& vpuTensorIteratorOutput = getVpuData(tensorIteratorOutput.get_node_shared_ptr());
//             VPU_THROW_UNLESS(vpuTensorIteratorOutput != nullptr, "Tensor Iterator's outputs must be parsed already");

//             const auto& loopEndOutput = vpuTensorIteratorOutput;

//             const auto& iterationInput = iteration.second;
//             VPU_THROW_UNLESS(getVpuData(iterationInput) == nullptr, "Tensor Iterator's body output data objects were not parsed yet");
//             auto loopEndInput = createData(iterationInput);

//             end_iteration_components.emplace(std::make_pair(loopEndOutputs.size(), rule), loopEndInputs.size());
//             loopEndInputs.push_back(loopEndInput);
//             loopEndOutputs.push_back(loopEndOutput);

//             // bindData(loopEndInput, iterationInput);
//         }

//         for (const auto& intermediateDataObject : intermediateDataObjects) {
//             const auto& tensorIteratorOutput = intermediateDataObject.first;
//             const auto& vpuTensorIteratorOutput = getVpuData(tensorIteratorOutput.get_node_shared_ptr());
//             VPU_THROW_UNLESS(vpuTensorIteratorOutput != nullptr, "Tensor Iterator's outputs must be parsed already");

//             auto loopEndOutput = vpuTensorIteratorOutput;
//             if (loopEndOutput->usage() == DataUsage::Output) {
//                 auto to_copy = model->addNewData(loopEndOutput->name() + "@copy-for-backedge", loopEndOutput->desc());
//                 _stageBuilder->addCopyStage(model, "copy-for-tensor-iterator-output", nullptr, to_copy, loopEndOutput, "copy for TI output");
//                 loopEndOutput = to_copy;
//             }

//             const auto& intermediateDataInput = intermediateDataObject.second;
//             VPU_THROW_UNLESS(getVpuData(intermediateDataInput) == nullptr, "Tensor Iterator's body output data objects were not parsed yet");

//             auto loopEndInput = createData(intermediateDataInput);

//             // to introduce shared data allocation edge later in Middle-End
//             loopEndOutput->attrs().set<Data>("end-shared-allocation", loopEndInput);

//             loopEndInputs.push_back(loopEndInput);
//             loopEndOutputs.push_back(loopEndOutput);

//             // bindData(loopEndInput, intermediateDataInput);
//         }

//         auto loopEnd = _stageBuilder->addLoopEndStage(model, tensorIterator->get_friendly_name() + "@LoopEnd", loopEndInputs, loopEndOutputs);
//         loopEnd->attrs().set("end-iteration-components", end_iteration_components);
//         if (batchIdx.hasValue()) {
//             loopEnd->attrs().set("batchId", batchIdx.get());
//         }

//         return loopEnd;
//     };

//     // Loop End must be introduced first to parse Tensor Iterator's body output data objects before parsing back-edge connections
//     auto loopEnd = introduceLoopEnd();
//     auto loopStart = introduceLoopStart();

//     // if (!tensorIterator->params.count(s_tripCountPort)) {
//     //     const auto iterationsCount = getNumIteration(*tensorIterator);
//     //     VPU_THROW_UNLESS(iterationsCount >= 0, "Encountered Tensor Iterator with iterations count equal to {}, but only non-negative values are supported",
//     //         iterationsCount);
//     //     loopStart->attrs().set<std::uint32_t>("iterations-count", static_cast<std::uint32_t>(iterationsCount));
//     //     loopEnd->attrs().set<std::uint32_t>("iterations-count", static_cast<std::uint32_t>(iterationsCount));
//     // }

//     // to allocate LoopEnd and LoopStart at the same time
//     loopStart->attrs().set<Stage>("loop-end", loopEnd);

//     // to be sure all loop's inputs are still alive during loop execution
//     // force they to be alive as long as loop's outputs
//     for (const auto& loopStartInput : loopStart->inputs()) {
//         model->addStageInput(loopEnd, loopStartInput);
//     }

//     for (const auto& bodyNode : body->get_ordered_ops()) {
//         if (ngraph::as_type_ptr<ngraph::opset4::Constant>(bodyNode)) {
//             // since Tensor Iterator's body is a kind of CNNNetwork it may has const data objects as inputs
//             // const data objects are hold by "Const" layers
//             // we don't need them during iteration and ignore the same way as compilation process of regular network
//             continue;
//         }

//         auto stageInputs = DataVector{};
//         auto stageOutputs = DataVector{};
//         for (const auto& input : bodyNode->inputs()) {
//             const auto& inputNode = input.get_node()->shared_from_this();
//             const auto vpuInput = isConst(inputNode) ? createConstData(inputNode) : getVpuData(inputNode);
//             VPU_THROW_UNLESS(vpuInput != nullptr,
//                 "Non-const input of a stage must be already parsed due to either topological order or as Loop Start's output");

//             stageInputs.push_back(vpuInput);

//             auto outputNode = input.get_source_output().get_node_shared_ptr();
//             auto output = getVpuData(outputNode);
//             if (output == nullptr) {
//                 output = createData(outputNode);
//                 // bindData(output, outputNode);
//             }
//             stageOutputs.push_back(output);

//         }

//         parseLayer(model, bodyNode, stageInputs, stageOutputs);
//     }
// }

// }  // namespace vpu
