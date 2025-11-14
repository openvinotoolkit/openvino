// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_remote_tensor.hpp"

#include <fstream>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::intel_npu;
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

void ZeroRemoteTensor::copy_file_data_to_level_zero_memory(const size_t size_to_read) {
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

    OPENVINO_ASSERT(size_to_read <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()),
                    "Cannot set offset ",
                    size_to_read,
                    " from ",
                    size_to_read,
                    ", because the value exceeds std::streamsize limit");

    std::streamoff offset = static_cast<std::streamoff>(_file_descriptor.value()._offset_in_bytes);

    std::ifstream fin(_file_descriptor.value()._file_path, std::ios::binary);

    fin.seekg(0, std::ios::end);
    std::streamoff file_size = fin.tellg();

    if (offset >= file_size) {
        OPENVINO_THROW("Offset is beyond the end of the file.");
    }

    fin.seekg(offset, std::ios::beg);

    _host_memory = ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                                    size_to_read,
                                                                    utils::STANDARD_PAGE_SIZE,
                                                                    _tensor_type == TensorType::INPUT ? true : false);
    _data = _host_memory->data();

    std::streamoff bytes_to_read = static_cast<std::streamoff>(size_to_read);
    fin.read(static_cast<char*>(_data), bytes_to_read);
    OPENVINO_ASSERT(fin.gcount() == bytes_to_read,
                    "Cannot read ",
                    bytes_to_read,
                    " bytes from ",
                    _file_descriptor.value()._file_path);
}

void ZeroRemoteTensor::allocate(const size_t bytes) {
    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF: {
        _host_memory =
            ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                             bytes,
                                                             utils::STANDARD_PAGE_SIZE,
                                                             _tensor_type == TensorType::INPUT ? true : false);
        _data = _host_memory->data();
        break;
    }
    case MemType::SHARED_BUF: {
        // set up the request to import the external memory handle
        _host_memory = ZeroMemPool::get_instance().import_shared_memory(_init_structs, _mem, bytes);
        _data = _host_memory->data();
        break;
    }
    case MemType::MMAPED_FILE: {
        // File_descriptor shall be set if mem_type is a mmaped file type.
        OPENVINO_ASSERT(_file_descriptor.has_value(),
                        "No parameter ",
                        file_descriptor.name(),
                        " found in parameters map");

        if (!_init_structs->isExternalMemoryStandardAllocationSupported()) {
            _logger.info("Importing mmaped memory isn't supported for this configuration. File data will be copied to "
                         "the level zero memory");
            copy_file_data_to_level_zero_memory(bytes);
            break;
        }

        if (_tensor_type == TensorType::OUTPUT) {
            // It is impossible to work on output today since ov::read_tensor_data opens the file in read-only mode.
            OPENVINO_THROW("Importing memory from a memory-mapped file is supported only for input tensors");
        } else if (_tensor_type == TensorType::BINDED) {
            _logger.warning("Importing memory from a memory-mapped file is supported only for input tensors");
            _tensor_type = TensorType::INPUT;
        }

        _mmap_tensor = ov::read_tensor_data(_file_descriptor.value()._file_path,
                                            _element_type,
                                            _shape,
                                            _file_descriptor.value()._offset_in_bytes,
                                            true);

        try {
            size_t aligned_size = utils::align_size_to_standard_page_size(bytes);
            _host_memory = ZeroMemPool::get_instance().import_standard_allocation_memory(
                _init_structs,
                _mmap_tensor.data(),
                aligned_size,
                _tensor_type == TensorType::INPUT ? true : false);
            _data = _host_memory->data();
        } catch (const ZeroMemException&) {
            _logger.info("Failed to import mmaped memory. File data will be copied to the level zero memory");
            _mmap_tensor = {};  // destroy memory if it couldn't be imported
            copy_file_data_to_level_zero_memory(bytes);
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
