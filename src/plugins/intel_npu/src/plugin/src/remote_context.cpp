// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_context.hpp"

#include "async_infer_request.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_execution_list.hpp"

using namespace ov::intel_npu;

namespace intel_npu {

namespace {

// Maps the requests the application holds to the plugin's requests and keeps them alive while they are in the list
class ExecutionList final : public IExecutionList {
public:
    ExecutionList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const void* queue_owner)
        : _list(init_structs, queue_owner) {}

    void append(const std::shared_ptr<ov::IAsyncInferRequest>& request) override {
        const auto async_request = std::dynamic_pointer_cast<AsyncInferRequest>(request);
        OPENVINO_ASSERT(async_request != nullptr, "The request was not created by the NPU plugin");
        const auto zero_request = std::dynamic_pointer_cast<ZeroInferRequest>(async_request->get_infer_request());
        OPENVINO_ASSERT(zero_request != nullptr, "The request is not supported by the execution list");

        _list.append(zero_request);
        _requests.push_back(request);
    }

    void prepare() override {
        _list.prepare();
    }

    void start_async() override {
        _list.start_async();
    }

    void wait() override {
        _list.wait();
    }

    void reset() override {
        _list.reset();
        _requests.clear();
    }

private:
    ZeroExecutionList _list;
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> _requests;
};

}  // namespace

RemoteContextImpl::RemoteContextImpl(const ov::SoPtr<IEngineBackend>& engineBackend,
                                     const ov::AnyMap& remote_properties)
    : _device_name("NPU") {
    if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
        OPENVINO_THROW("Level zero backend is not found!");
    }

    _init_structs = engineBackend->getInitStructs();

    _properties = {l0_context(engineBackend->getContext())};

    if (!remote_properties.empty()) {
        _mem_type_object = zeroUtils::extract_object(remote_properties, mem_type);
        _tensor_type_object = zeroUtils::extract_object(remote_properties, tensor_type);
        _mem_handle_object = zeroUtils::extract_object(remote_properties, mem_handle);
        _file_descriptor_object = zeroUtils::extract_object(remote_properties, file_descriptor);
    }
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return _properties;
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type,
                                                              const ov::Shape& shape,
                                                              const ov::AnyMap& params) {
    // Local remote properties
    std::optional<ov::intel_npu::MemType> mem_type_object = std::nullopt;
    std::optional<ov::intel_npu::TensorType> tensor_type_object = std::nullopt;
    std::optional<void*> mem_handle_object = std::nullopt;
    std::optional<ov::intel_npu::FileDescriptor> file_descriptor_object = std::nullopt;

    if (!params.empty()) {
        // Save local remote properties.
        mem_type_object = zeroUtils::extract_object(params, mem_type);
        tensor_type_object = zeroUtils::extract_object(params, tensor_type);
        mem_handle_object = zeroUtils::extract_object(params, mem_handle);
        file_descriptor_object = zeroUtils::extract_object(params, file_descriptor);
    }

    // Merge local remote properties with global remote properties.
    if (!mem_type_object.has_value() && _mem_type_object.has_value()) {
        mem_type_object = _mem_type_object;
    }
    if (!tensor_type_object.has_value() && _tensor_type_object.has_value()) {
        tensor_type_object = _tensor_type_object;
    }
    if (!mem_handle_object.has_value() && _mem_handle_object.has_value()) {
        mem_handle_object = _mem_handle_object;
    }
    if (!file_descriptor_object.has_value() && _file_descriptor_object.has_value()) {
        file_descriptor_object = _file_descriptor_object;
    }

    // Mem_type shall be set if any other property is set.
    if (!mem_type_object.has_value() &&
        (mem_handle_object.has_value() || tensor_type_object.has_value() || file_descriptor_object.has_value())) {
        OPENVINO_THROW("Parameter ", mem_type.name(), " must be set");
    }

    if (!mem_type_object.has_value()) {
        return {std::make_shared<ZeroRemoteTensor>(get_this_shared_ptr(), _init_structs, type, shape)};
    }

    // Mem_handle shall be set if mem_type is a shared memory type or a CPU virtual address.
    if ((mem_type_object.value() == MemType::SHARED_BUF || mem_type_object.value() == MemType::CPU_VA) &&
        !mem_handle_object.has_value()) {
        OPENVINO_THROW("No parameter ", mem_handle.name(), " found in parameters map");
    }

    return {std::make_shared<ZeroRemoteTensor>(get_this_shared_ptr(),
                                               _init_structs,
                                               type,
                                               shape,
                                               tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED),
                                               mem_type_object.value_or(ov::intel_npu::MemType::L0_INTERNAL_BUF),
                                               mem_handle_object.value_or(nullptr),
                                               file_descriptor_object)};
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    if (type == ov::element::string) {
        return ov::IRemoteContext::create_host_tensor(type, shape);
    }
    return {std::make_shared<ZeroHostTensor>(get_this_shared_ptr(),
                                             _init_structs,
                                             type,
                                             shape,
                                             _tensor_type_object.value_or(ov::intel_npu::TensorType::BINDED))};
}

const std::string& RemoteContextImpl::get_device_name() const {
    return _device_name;
}

std::shared_ptr<IExecutionList> RemoteContextImpl::create_execution_list(const void* queue_owner) {
    return std::make_shared<ExecutionList>(_init_structs, queue_owner);
}

std::shared_ptr<ov::IRemoteContext> RemoteContextImpl::get_this_shared_ptr() {
    return shared_from_this();
}

}  // namespace intel_npu
