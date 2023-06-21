// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

#include <ie_parallel.hpp>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <string>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "cpp_interfaces/plugin_itt.hpp"
#include "debug.h"
#include "ie_algorithm.hpp"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_compound_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_preprocess.hpp"
#include "ie_remote_context.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

IInferRequestInternal::~IInferRequestInternal() {}

IInferRequestInternal::IInferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs)
    :  // We should copy maps since they can be overriden in SetBlob with preprocess
      _networkInputs{copyInfo(networkInputs)},
      _networkOutputs{copyInfo(networkOutputs)} {}

IInferRequestInternal::IInferRequestInternal(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs)
    : _parameters(inputs),
      _results(outputs) {
    const auto& create_old_data = [](const ov::Output<const ov::Node>& output) -> InferenceEngine::DataPtr {
        auto name = ov::op::util::get_ie_output_name(output);
        auto shape = output.get_partial_shape();
        auto rank = shape.rank().is_static() ? shape.rank().get_length() : -1;
        SizeVector dims(1, 0);
        if (shape.is_static()) {
            dims = output.get_shape();
        } else if (rank >= 0) {
            dims = SizeVector(rank, 0);
        }
        for (const auto& dim : shape) {
            if (dim.is_static() && dim.get_length() == 0)
                IE_THROW() << name << " has zero dimension which is not allowed";
        }
        const Layout rankLayout = rank < 0 ? Layout::BLOCKED : TensorDesc::getLayoutByRank(rank);
        const auto precision = InferenceEngine::details::convertPrecision(output.get_element_type());
        return std::make_shared<Data>(name, TensorDesc{precision, dims, rankLayout});
    };
    const auto& create_old_input_data =
        [create_old_data](const ov::Output<const ov::Node>& output) -> InferenceEngine::InputInfo::Ptr {
        auto info = std::make_shared<InferenceEngine::InputInfo>();
        info->setInputData(create_old_data(output));
        return info;
    };

    for (const auto& param : _parameters) {
        const auto& input = create_old_input_data(param->output(0));
        input->setName(param->get_friendly_name());
        _networkInputs[input->name()] = input;
    }

    for (const auto& result : _results) {
        auto input = result->input_value(0);
        const auto& output = create_old_data(ov::Output<const ov::Node>(input.get_node(), input.get_index()));
        _networkOutputs[output->getName()] = output;
    }
}

void IInferRequestInternal::Infer() {
    checkBlobs();
    InferImpl();
}

void IInferRequestInternal::InferImpl() {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::Cancel() {
    IE_THROW(NotImplemented);
}

std::map<std::string, InferenceEngineProfileInfo> IInferRequestInternal::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<const ov::Node> IInferRequestInternal::findInputByNodeName(const std::string& name) const {
    for (const auto& input : GetInputs()) {
        if (input->get_friendly_name() == name)
            return input;
    }
    return nullptr;
}

std::shared_ptr<const ov::Node> IInferRequestInternal::findOutputByNodeName(const std::string& name) const {
    for (const auto& output : GetOutputs()) {
        if (output->input_value(0).get_node()->get_friendly_name() == name)
            return output;
    }
    return nullptr;
}

void IInferRequestInternal::SetBlob(const std::string& name, const Blob::Ptr& userBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::Plugin, "SetBlob");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!userBlob)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const auto input = findInputByNodeName(name);
    const auto output = findOutputByNodeName(name);

    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed = userBlob->is<RemoteBlob>();
    if (!compoundBlobPassed && !remoteBlobPassed && userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    if (userBlob->size() == 0 && !((input && input->get_output_partial_shape(0).is_dynamic()) ||
                                   (output && output->get_output_partial_shape(0).is_dynamic()))) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }
    const bool isInputDynamic = input && input->get_output_partial_shape(0).is_dynamic();
    const bool isOutputDynamic = output && output->get_input_partial_shape(0).is_dynamic();

    size_t dataSize = userBlob->size();
    if (isInput) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user input precision";
        }

        auto& devBlob = _deviceInputs[name];
        const bool preProcRequired = preProcessingRequired(foundInput, userBlob, devBlob);
        if (compoundBlobPassed && !preProcRequired) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            addInputPreProcessingFor(name, userBlob, devBlob ? devBlob : _inputs[name]);
        } else {
            size_t inputSize = foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                   ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
                                   : 1;
            if (!isInputDynamic && dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size (" << dataSize << "!=" << inputSize
                           << ").";
            }
            _inputs[name] = userBlob;
            devBlob = userBlob;
        }
        _batched_inputs.erase(name);
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                ? details::product(foundOutput->getTensorDesc().getDims())
                                : 1;
        if (!isOutputDynamic && dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user output precision";
        }
        // ilavreno: this condition is valid for most plugins except MYRIAD
        // it is able to perform layout conversion for output blob dynamically
        // if (foundOutput->getLayout() != userBlob->getTensorDesc().getLayout()) {
        //     IE_THROW(ParameterMismatch) << "Failed to set Blob with layout not corresponding to user output layout";
        // }
        _outputs[name] = userBlob;
    }
}

void IInferRequestInternal::SetBlobs(const std::string& name, const std::vector<Blob::Ptr>& blobs) {
    if (blobs.size() == 1) {
        SetBlob(name, blobs[0]);
        return;
    }

    bool all_memory = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& item) {
        return item && item->is<MemoryBlob>() && !item->is<RemoteBlob>();
    });
    OPENVINO_ASSERT(all_memory,
                    "set_input_tensors/set_tensors error. Default implementation support only local memory tensors");

    checkBlobsForBatch(name, blobs);

    SetBlobsImpl(name, std::make_shared<BatchedBlob>(blobs));
}

void IInferRequestInternal::SetBlobsImpl(const std::string& name, const BatchedBlob::Ptr& batched_blob) {
    IE_THROW(NotImplemented) << "set_input_tensors/set_tensors are not supported by this plugin";
}

void IInferRequestInternal::checkBlobsForBatch(const std::string& name, const std::vector<Blob::Ptr>& blobs) {
    OPENVINO_ASSERT(!blobs.empty(),
                    "set_input_tensors/set_tensors can't be called with empty blobs for input '",
                    name,
                    "'");
    OPENVINO_ASSERT(blobs.size() != 1,
                    "Internal error (plugin): checkBlobsForBatch is not allowed to have only one blob inside batch "
                    "for input '",
                    name,
                    "'");

    std::shared_ptr<const ov::op::v0::Parameter> param;
    const auto& inputs = GetInputs();
    for (const auto& input : inputs) {
        if (auto p = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(input)) {
            if (name == p->get_friendly_name()) {
                param = p;
                break;
            }
        }
    }
    OPENVINO_ASSERT(param, "set_input_tensors/set_tensors error. Parameter '", name, "' is not found");
    OPENVINO_ASSERT(ov::layout::has_batch(param->get_layout()),
                    "set_input_tensors/set_tensors can be used only for inputs with N(batch) dimension"
                    " 'layout' defined. Current layout for '",
                    name,
                    "' is ",
                    param->get_layout().to_string());
    auto batch_idx = ov::layout::batch_idx(param->get_layout());
    if (batch_idx < 0) {
        batch_idx += static_cast<int64_t>(blobs[0]->getTensorDesc().getDims().size());
    }
    OPENVINO_ASSERT(batch_idx == 0,
                    "set_input_tensors/set_tensors is not currently supported for batch dimension index ",
                    batch_idx,
                    " != 0");
    std::for_each(blobs.begin(), blobs.end(), [&batch_idx](const Blob::Ptr& item) {
        OPENVINO_ASSERT(item->getTensorDesc().getDims()[batch_idx] == 1,
                        "set_input_tensors/set_tensors. Tensors shall represent one item in a batch, ",
                        item->getTensorDesc().getDims()[batch_idx],
                        " provided");
    });
    auto blobs_size = static_cast<int>(blobs.size());
    if (param->get_partial_shape().rank().is_static()) {
        OPENVINO_ASSERT(batch_idx >= 0 && batch_idx < param->get_partial_shape().rank().get_length(),
                        "set_input_tensors/set_tensors error. Layout ",
                        param->get_layout().to_string(),
                        " is incorrect for operation with name '",
                        name,
                        "' with shape ",
                        param->get_partial_shape());
        auto batch = param->get_partial_shape()[batch_idx];

        OPENVINO_ASSERT(batch.is_dynamic() || batch.get_length() == blobs_size,
                        "set_input_tensors/set_tensors error. Input shape ",
                        param->get_partial_shape(),
                        "batch ",
                        batch,
                        "doesn't match with total blobs count: ",
                        blobs_size);
    }

    // In future consider checking if blobs point to contiguous range of memory and use single 'SetBlob' instead
    auto tmp_desc = blobs[0]->getTensorDesc();
    tmp_desc.getDims()[batch_idx] = blobs_size;
    auto blockingDims = tmp_desc.getBlockingDesc().getBlockDims();
    blockingDims[batch_idx] = blobs_size;
    auto blockingDesc = BlockingDesc(blockingDims, tmp_desc.getBlockingDesc().getOrder());
    auto batched_desc = InferenceEngine::TensorDesc(tmp_desc.getPrecision(), tmp_desc.getDims(), blockingDesc);
    auto desc_to_string = [](const TensorDesc& desc) {
        std::stringstream s;
        s << "{ " << desc.getLayout() << " " << desc.getPrecision().name();
        s << "dim=(";
        for (const auto& d : desc.getDims()) {
            s << " " << d;
        }
        s << " ) }";
        return s.str();
    };
    for (const auto& item : blobs) {
        auto item_desc = item->getTensorDesc();
        item_desc.getDims()[batch_idx] = batched_desc.getDims()[batch_idx];
        OPENVINO_ASSERT(item_desc.getDims() == batched_desc.getDims() &&
                            item_desc.getLayout() == batched_desc.getLayout() &&
                            item_desc.getPrecision() == batched_desc.getPrecision() &&
                            item_desc.getBlockingDesc().getOrder() == batched_desc.getBlockingDesc().getOrder(),
                        "set_input_tensors/set_tensors error. Blob ",
                        desc_to_string(item_desc),
                        " is not compatible with batched blob ",
                        desc_to_string(batched_desc));
    }
}

void IInferRequestInternal::convertBatchedInputBlob(const std::string& name, const BatchedBlob::Ptr& batched_blob) {
    auto tmp_desc = batched_blob->getBlob(0)->getTensorDesc();
    tmp_desc.getDims()[0] = batched_blob->size();
    auto blockingDims = tmp_desc.getBlockingDesc().getBlockDims();
    blockingDims[0] = batched_blob->size();
    auto blockingDesc = BlockingDesc(blockingDims, tmp_desc.getBlockingDesc().getOrder());
    auto batched_desc = InferenceEngine::TensorDesc(tmp_desc.getPrecision(), tmp_desc.getDims(), blockingDesc);
    std::shared_ptr<RemoteContext> remote_context;
    MemoryBlob::Ptr mem_blob;
    try {
        auto net = getPointerToExecutableNetworkInternal();
        if (net) {
            remote_context = net->GetContext();
        }
    } catch (const InferenceEngine::NotImplemented&) {
    }
    if (remote_context) {
        mem_blob = remote_context->CreateHostBlob(batched_desc);
    } else {
        mem_blob = std::dynamic_pointer_cast<MemoryBlob>(make_blob_with_precision(batched_desc));
    }
    OPENVINO_ASSERT(mem_blob, "Internal error - can't create host memory blob");
    mem_blob->allocate();
    auto ptr = mem_blob->wmap();

    // Perform memory copy
    InferenceEngine::parallel_for(batched_blob->size(), [&](size_t i) {
        const auto& blob = as<MemoryBlob>(batched_blob->getBlob(i));
        OPENVINO_ASSERT(mem_blob, "Internal error - can't cast blob ", i, " to MemoryBlob");
        const auto& blob_desc = blob->getTensorDesc().getBlockingDesc();
        bool offsets_0 = std::all_of(blob_desc.getOffsetPaddingToData().begin(),
                                     blob_desc.getOffsetPaddingToData().end(),
                                     [](size_t dim) {
                                         return dim == 0;
                                     });
        OPENVINO_ASSERT(offsets_0,
                        "set_tensors/set_input_tensors - default combining is not supported for "
                        "ROI tensors. All tensors offsets shall be 0");
        OPENVINO_ASSERT(mem_blob->getTensorDesc().getBlockingDesc().getOrder() == blob_desc.getOrder(),
                        "set_tensors/set_input_tensors - default combining is not supported for "
                        "ROI tensors. Axis order shall be default");
        OPENVINO_ASSERT(mem_blob->getTensorDesc().getBlockingDesc().getStrides() == blob_desc.getStrides(),
                        "set_tensors/set_input_tensors - default combining is not supported for "
                        "ROI tensors. Input blobs shall have default strides set");
        memcpy(ptr.as<uint8_t*>() + i * blob->byteSize(),
               blob->rmap().as<uint8_t*>() +
                   blob->getTensorDesc().getBlockingDesc().getOffsetPadding() * blob->element_size(),
               blob->byteSize());
    });
    SetBlob(name, mem_blob);
}

void IInferRequestInternal::convertBatchedInputBlobs() {
    auto batched_copy = _batched_inputs;
    for (const auto& item : batched_copy) {
        convertBatchedInputBlob(item.first, item.second);
    }
    _batched_inputs = batched_copy;
}

Blob::Ptr IInferRequestInternal::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::Plugin, "GetBlob");
    Blob::Ptr data;
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    const SizeVector oneVector = {1};
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        const auto input = findInputByNodeName(name);
        const bool isInputDynamic = input && input->get_output_partial_shape(0).is_dynamic();
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            const auto& dims = foundInput->getTensorDesc().getDims();
            if (isInputDynamic)
                checkBlob(data, name, true);
            else
                checkBlob(data, name, true, foundInput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);

            auto& devBlob = _deviceInputs[name];
            if (preProcessingRequired(foundInput, data, devBlob)) {
                // if no devBlob, performs inplace
                addInputPreProcessingFor(name, data, devBlob ? devBlob : _inputs[name]);
            }
        }
    } else {
        const auto output = findOutputByNodeName(name);
        const bool isOutputDynamic = output && output->get_output_partial_shape(0).is_dynamic();
        data = _outputs[name];
        const auto& dims = foundOutput->getTensorDesc().getDims();
        if (isOutputDynamic)
            checkBlob(data, name, false);
        else
            checkBlob(data, name, false, foundOutput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    }
    return data;
}

BatchedBlob::Ptr IInferRequestInternal::GetBlobs(const std::string& name) {
    if (_batched_inputs.count(name)) {
        return _batched_inputs.at(name);
    }
    return nullptr;
}

const PreProcessInfo& IInferRequestInternal::GetPreProcess(const std::string& name) const {
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        return foundInput->getPreProcess();
    } else {
        IE_THROW() << "Output blob can't have pre-processing";
    }
}

std::vector<std::shared_ptr<IVariableStateInternal>> IInferRequestInternal::QueryState() {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::StartAsync() {
    checkBlobs();
    StartAsyncImpl();
}

void IInferRequestInternal::StartAsyncImpl() {
    IE_THROW(NotImplemented);
}

StatusCode IInferRequestInternal::Wait(int64_t millis_timeout) {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::SetCallback(Callback callback) {
    _callback = std::move(callback);
}

void IInferRequestInternal::execDataPreprocessing(InferenceEngine::BlobMap& preprocessedBlobs, bool serial) {
    for (auto& input : preprocessedBlobs) {
        // If there is a pre-process entry for an input then it must be pre-processed
        // using preconfigured resize algorithm.
        auto it = _preProcData.find(input.first);
        if (it != _preProcData.end()) {
            it->second->execute(input.second, _networkInputs[input.first]->getPreProcess(), serial, -1);
        }
    }
}

bool IInferRequestInternal::findInputAndOutputBlobByName(const std::string& name,
                                                         InputInfo::Ptr& foundInput,
                                                         DataPtr& foundOutput) const {
    foundInput = nullptr;
    foundOutput = nullptr;
    if (_networkOutputs.empty()) {
        IE_THROW() << "Internal error: network outputs is not set";
    }
    auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                       std::end(_networkInputs),
                                       [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                           return pair.first == name;
                                       });
    auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                        std::end(_networkOutputs),
                                        [&](const std::pair<std::string, DataPtr>& pair) {
                                            return pair.first == name;
                                        });
    bool retVal;

    if (foundInputPair != std::end(_networkInputs)) {
        foundInput = foundInputPair->second;
        retVal = true;
    } else if (foundOutputPair != std::end(_networkOutputs)) {
        foundOutput = foundOutputPair->second;
        retVal = false;
    } else {
        IE_THROW(NotFound) << "Failed to find input or output with name: \'" << name << "\'";
    }
    return retVal;
}

void IInferRequestInternal::checkBlob(const Blob::Ptr& blob,
                                      const std::string& name,
                                      bool isInput,
                                      const SizeVector& refDims) const {
    std::string bType = isInput ? "Input" : "Output";
    std::string sType = isInput ? "input" : "output";
    std::string strNotAllocated(bType + " data was not allocated.");
    std::string strNotMatched("The " + sType + " blob size is not equal to the network " + sType + " size");

    if (!blob) {
        IE_THROW(NotAllocated) << strNotAllocated;
    }
    size_t refSize;
    bool isDynamic = false;
    if (refDims.empty()) {
        SizeVector dims;
        if (isInput) {
            auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                               std::end(_networkInputs),
                                               [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                                   return pair.first == name;
                                               });
            if (foundInputPair == std::end(_networkInputs)) {
                IE_THROW(NotFound) << "Failed to find input with name: \'" << name << "\'";
            }
            const auto input = findInputByNodeName(name);
            isDynamic = input && input->get_output_partial_shape(0).is_dynamic();
            dims = foundInputPair->second->getTensorDesc().getDims();
            refSize = foundInputPair->second->getTensorDesc().getLayout() != SCALAR ? details::product(dims) : 1;
        } else {
            auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                                std::end(_networkOutputs),
                                                [&](const std::pair<std::string, DataPtr>& pair) {
                                                    return pair.first == name;
                                                });
            if (foundOutputPair == std::end(_networkOutputs)) {
                IE_THROW(NotFound) << "Failed to find output with name: \'" << name << "\'";
            }
            const auto output = findOutputByNodeName(name);
            isDynamic = output && output->get_output_partial_shape(0).is_dynamic();
            ngraph::PartialShape blobPartialShape(blob->getTensorDesc().getDims());
            if (output && output->get_output_partial_shape(0).compatible(blobPartialShape)) {
                dims = blob->getTensorDesc().getDims();
            } else {
                // TODO: it is strange to request tensor desc from data when the shapes are not compatible, probably we
                // need to immediately throw here
                dims = foundOutputPair->second->getTensorDesc().getDims();
            }
            refSize = foundOutputPair->second->getTensorDesc().getLayout() != SCALAR ? details::product(dims) : 1;
        }
    } else {
        refSize = details::product(refDims);
    }

    if (!isDynamic && refSize != blob->size()) {
        IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }
    const bool remoteBlobPassed = blob->is<RemoteBlob>();
    if (!remoteBlobPassed && blob->buffer() == nullptr)
        IE_THROW() << strNotAllocated;
}

void IInferRequestInternal::checkBlobs() {
    for (auto const& input : _inputs) {
        checkBlob(input.second, input.first, true);
    }
    for (auto const& output : _outputs) {
        checkBlob(output.second, output.first, false);
    }
}

void IInferRequestInternal::setPointerToExecutableNetworkInternal(
    const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork) {
    _exeNetwork = exeNetwork;
}

std::shared_ptr<IExecutableNetworkInternal> IInferRequestInternal::getPointerToExecutableNetworkInternal() const {
    return _exeNetwork;
}

void IInferRequestInternal::setPointerToSo(const std::shared_ptr<void>& so) {
    _so = so;
}

std::shared_ptr<void> IInferRequestInternal::getPointerToSo() const {
    return _so;
}

bool IInferRequestInternal::preProcessingRequired(const InputInfo::Ptr& info,
                                                  const Blob::Ptr& userBlob,
                                                  const Blob::Ptr& deviceBlob) {
    // pre-processing is required if:
    // 1. resize algorithm is specified (resize required)
    // 2. color format specified:
    // 2.a. color format is not equal to network's expected (color conversion required)
    // 2.b. network's layout != blob's layout (reorder required)
    // 3. precision conversion is required

    const auto& preProcessInfo = info->getPreProcess();
    const auto inputColorFormat = preProcessInfo.getColorFormat();
    // FIXME: support other network's input formats once the API is ready. Assuming input is in
    // the BGR format by default
    const auto networkColorFormat = ColorFormat::BGR;
    const bool colorFormatSpecified = inputColorFormat != ColorFormat::RAW;

    auto blob_layout = [](const Blob::Ptr& b) {
        return b->getTensorDesc().getLayout();
    };
    auto blob_prec = [](const Blob::Ptr& b) {
        return b->getTensorDesc().getPrecision();
    };

    auto dst_layout = deviceBlob ? blob_layout(deviceBlob) : info->getLayout();
    auto dst_prec = deviceBlob ? blob_prec(deviceBlob) : info->getPrecision();

    // FIXME: remove the first part to allow any needed conversion?
    const bool need_layout_conv = (colorFormatSpecified || deviceBlob) && (blob_layout(userBlob) != dst_layout);

    return preProcessInfo.getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE ||
           (colorFormatSpecified && inputColorFormat != networkColorFormat) || need_layout_conv ||
           (blob_prec(userBlob) != dst_prec);
}

void IInferRequestInternal::addInputPreProcessingFor(const std::string& name,
                                                     const Blob::Ptr& from,
                                                     const Blob::Ptr& to) {
    auto ppDataIt = _preProcData.find(name);
    if (ppDataIt == _preProcData.end()) {
        ppDataIt = (_preProcData.emplace(name, CreatePreprocDataHelper())).first;
    }

    auto& preproc_ptr = ppDataIt->second;
    preproc_ptr->isApplicable(from, to);
    // Stores the given blob as ROI blob. It will be used to fill in network input
    // during pre-processing
    preproc_ptr->setRoiBlob(from);
}

void* IInferRequestInternal::GetUserData() noexcept {
    return _userData;
}

void IInferRequestInternal::SetUserData(void* userData) noexcept {
    _userData = userData;
}

void IInferRequestInternal::setModelInputsOutputs(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                  const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    _parameters = inputs;
    _results = outputs;
}

const std::vector<std::shared_ptr<const ov::Node>>& IInferRequestInternal::GetInputs() const {
    return _parameters;
}

const std::vector<std::shared_ptr<const ov::Node>>& IInferRequestInternal::GetOutputs() const {
    return _results;
}
}  // namespace InferenceEngine
