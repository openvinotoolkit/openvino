// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_remote_tensor.hpp"

#include <ze_api.h>
#include <ze_mem_import_system_memory_ext.h>

#include <fstream>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::intel_npu;

namespace {

static uint32_t to_alloc_flag(TensorType tensor_type) {
    if (tensor_type == TensorType::INPUT) {
        return ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    }
    return 0;
}

}  // namespace

namespace intel_npu {

ZeroRemoteTensor::ZeroRemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                   const ov::element::Type& element_type,
                                   const ov::Shape& shape,
                                   TensorType tensor_type,
                                   MemType mem_type,
                                   const void* mem,
                                   const std::optional<ov::intel_npu::FileDescriptor>& file_descriptor)
    : _context(context),
      _init_structs(init_structs),
      _element_type(element_type),
      _shape(shape),
      _capacity(shape),
      _logger("ZeroRemoteContext", Logger::global().level()),
      _tensor_type(tensor_type),
      _mem_type(mem_type),
      _file_descriptor(file_descriptor),
      _mem(mem) {
    OPENVINO_ASSERT(shape_size(_shape) != 0);
    OPENVINO_ASSERT(_element_type.is_static());

    const auto byte_size = ov::util::get_memory_size_safe(element_type, shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", element_type, " and shape: ", shape);

    ze_device_external_memory_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
    auto res = zeDeviceGetExternalMemoryProperties(_init_structs->getDevice(), &desc);
    if (res == ZE_RESULT_SUCCESS) {
#ifdef _WIN32
        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) {
            _external_memory_support = true;
        }
#else
        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) {
            _external_memory_support = true;
        }
#endif

        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_STANDARD_ALLOCATION) {
            _mmaped_file_support = true;
        }
    }

    allocate(*byte_size);
}

const ov::element::Type& ZeroRemoteTensor::get_element_type() const {
    return _element_type;
}

const ov::Shape& ZeroRemoteTensor::get_shape() const {
    return _shape;
}

const ov::Strides& ZeroRemoteTensor::get_strides() const {
    return _strides;
}

const ov::AnyMap& ZeroRemoteTensor::get_properties() const {
    return _properties;
}

void ZeroRemoteTensor::set_shape(ov::Shape new_shape) {
    if (_shape == new_shape) {
        return;
    }

    _shape = std::move(new_shape);

    if (ov::shape_size(_shape) > ov::shape_size(_capacity)) {
        OPENVINO_THROW("Cannot set a new bigger shape to this tensor.");
    }

    _strides.clear();
    update_strides();
}

void ZeroRemoteTensor::update_strides() {
    if (_element_type.bitwidth() < 8) {
        return;
    }

    auto& shape = get_shape();
    if (_strides.empty() && !shape.empty()) {
        _strides.resize(shape.size());
        _strides.back() = shape.back() == 0 ? 0 : _element_type.size();
        std::transform(shape.crbegin(),
                       shape.crend() - 1,
                       _strides.rbegin(),
                       _strides.rbegin() + 1,
                       std::multiplies<size_t>());
    }
}

const std::string& ZeroRemoteTensor::get_device_name() const {
    return _context->get_device_name();
}

std::shared_ptr<ov::IRemoteContext> ZeroRemoteTensor::get_context() const {
    return _context;
}

ZeroRemoteTensor::~ZeroRemoteTensor() {
    auto res = deallocate();
    if (!res) {
        _logger.error("ZeroRemoteTensor failed to free the memory");
    }
}

bool ZeroRemoteTensor::deallocate() noexcept {
    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF:
    case MemType::SHARED_BUF:
    case MemType::MMAPED_FILE: {
        if (_data) {
            auto result = zeMemFree(_init_structs->getContext(), _data);
            if (ZE_RESULT_SUCCESS != result) {
                if (ZE_RESULT_ERROR_UNINITIALIZED == result) {
                    _logger.warning("ZeroRemoteTensor failed to free memory; Level zero context was already destroyed "
                                    "and memory was already released by the driver.");
                } else {
                    _logger.error("zeMemFree failed %#X", uint64_t(result));
                    return false;
                }
            }

            _data = nullptr;
        }

        return true;
    }
    default:
        return false;
    }
}

void ZeroRemoteTensor::copy_file_data_to_level_zero_memory() {
    if (!_file_descriptor.has_value()) {
        OPENVINO_THROW("No parameter ", file_descriptor.name(), " found in parameters map");
    }

    OPENVINO_ASSERT(
        _file_descriptor.value()._offset_in_bytes <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()),
        "Cannot set offset ",
        _file_descriptor.value()._offset_in_bytes,
        " from ",
        _file_descriptor.value()._file_path,
        ", because the value exceeds std::streamsize limit");

    std::streamoff offset = static_cast<std::streamoff>(_file_descriptor.value()._offset_in_bytes);

    std::ifstream fin(_file_descriptor.value()._file_path, std::ios::binary);

    fin.seekg(0, std::ios::end);
    std::streamoff file_size = fin.tellg();

    if (offset >= file_size) {
        OPENVINO_THROW("Offset is beyond the end of the file.");
    }

    std::streamoff size_to_read = file_size - offset;
    fin.seekg(offset, std::ios::beg);

    size_t aligned_size = utils::align_size_to_standard_page_size(size_to_read);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, to_alloc_flag(_tensor_type)};
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeMemAllocHost",
        zeMemAllocHost(_init_structs->getContext(), &desc, aligned_size, utils::STANDARD_PAGE_SIZE, &_data));

    fin.read(static_cast<char*>(_data), size_to_read);
    OPENVINO_ASSERT(fin.gcount() == size_to_read,
                    "Cannot read ",
                    size_to_read,
                    " bytes from ",
                    _file_descriptor.value()._file_path);
}

void ZeroRemoteTensor::allocate(const size_t bytes) {
    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF: {
        size_t size = utils::align_size_to_standard_page_size(bytes);

        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, to_alloc_flag(_tensor_type)};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, size, utils::STANDARD_PAGE_SIZE, &_data));
        break;
    }
    case MemType::SHARED_BUF: {
        if (!_external_memory_support) {
            OPENVINO_THROW("Remote tensor functionality is not supported with this driver version");
        }

        // set up the request to import the external memory handle
#ifdef _WIN32
        // in the case of the Windows platform memory is locked by the D3D12 memory management - using zeMemAllocDevice
        // to import memory
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(_mem),
                                                                  nullptr};
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, 0, 0};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocDevice",
                                    zeMemAllocDevice(_init_structs->getContext(),
                                                     &desc,
                                                     bytes,
                                                     utils::STANDARD_PAGE_SIZE,
                                                     _init_structs->getDevice(),
                                                     &_data));
#else
        // in the case of Linux platforms memory could be changed after allocation - using zeMemAllocHost for importing
        // memory
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(_mem))};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, bytes, utils::STANDARD_PAGE_SIZE, &_data));
#endif
        break;
    }
    case MemType::MMAPED_FILE: {
        // File_descriptor shall be set if mem_type is a mmaped file type.
        if (!_file_descriptor.has_value()) {
            OPENVINO_THROW("No parameter ", file_descriptor.name(), " found in parameters map");
        }

        if (!_mmaped_file_support) {
            _logger.info("Importing mmaped memory isn't supported for this configuration. File data will be copied to "
                         "the level zero memory");
            copy_file_data_to_level_zero_memory();
            break;
        }

        if (_tensor_type == TensorType::OUTPUT) {
            _logger.info("Importing mmaped memory isn't supported for output tensors. File data will be copied to the "
                         "level zero memory");

            copy_file_data_to_level_zero_memory();
            break;
        } else if (_tensor_type == TensorType::BINDED) {
            _logger.warning("Importing memory from a memory-mapped file is supported only for input tensors");
            _tensor_type = TensorType::INPUT;
        }

        // memory map the file
        _mmap_tensor = ov::read_tensor_data(_file_descriptor.value()._file_path,
                                            ov::element::u8,
                                            ov::PartialShape::dynamic(1),
                                            _file_descriptor.value()._offset_in_bytes,
                                            true);

        size_t aligned_size = utils::align_size_to_standard_page_size(_mmap_tensor.get_byte_size());

        _ze_external_memory_import_system_memory_t memory_import = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_SYSTEM_MEMORY,
            nullptr,
            _mmap_tensor.data(),
            aligned_size};

        uint32_t flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, flag};

        auto result =
            zeMemAllocHost(_init_structs->getContext(), &desc, aligned_size, utils::STANDARD_PAGE_SIZE, &_data);

        if (result != ZE_RESULT_SUCCESS) {
            _logger.info("Failed to import mmaped memory. File data will be copied to the level zero memory");
            _mmap_tensor = {};  // destroy memory if it couldn't be imported
            copy_file_data_to_level_zero_memory();
        }
        break;
    }
    default:
        _data = nullptr;
    }

    update_properties();
    update_strides();
}

bool ZeroRemoteTensor::is_allocated() const noexcept {
    return _data != nullptr;
}

void ZeroRemoteTensor::update_properties() {
    OPENVINO_ASSERT(is_allocated(), "Can't initialize ZeroRemoteTensor parameters as memory was not allocated");

    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF:
    case MemType::MMAPED_FILE:
        _properties = {mem_type(_mem_type), mem_handle(_data), tensor_type(_tensor_type)};

        break;
    case MemType::SHARED_BUF:
        _properties = {mem_type(_mem_type), mem_handle(_data)};

        break;
    default:
        OPENVINO_THROW("Unsupported object type ", static_cast<int>(_mem_type));
    }
}

void* ZeroRemoteTensor::get_original_memory() const {
    return _data;
}

ze_context_handle_t ZeroRemoteTensor::get_zero_context_handle() const {
    return _init_structs->getContext();
}

}  // namespace intel_npu
