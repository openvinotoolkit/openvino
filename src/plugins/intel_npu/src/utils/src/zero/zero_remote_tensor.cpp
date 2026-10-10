// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_remote_tensor.hpp"

#include <fstream>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::intel_npu;

namespace {

void validate_roi_bounds(const ov::ITensor& tensor, size_t offset, const ov::Shape& roi_shape) {
    tensor.get_strides_for_shape(roi_shape);
    const auto& owner_shape = tensor.get_shape();
    const auto& owner_strides = tensor.get_strides();
    if (ov::shape_size(roi_shape) == 0) {
        return;
    }
    OPENVINO_ASSERT(ov::shape_size(owner_shape) != 0, "Cannot copy a nonempty ROI from an empty tensor.");

    size_t remaining_offset = offset;
    for (size_t i = 0; i < owner_shape.size(); ++i) {
        OPENVINO_ASSERT(owner_strides[i] != 0, "Nonempty tensor strides must not be zero.");
        const auto begin = remaining_offset / owner_strides[i];
        remaining_offset %= owner_strides[i];
        const auto trailing_axis = owner_shape.size() - 1 - i;
        const auto dim = trailing_axis < roi_shape.size() ? roi_shape[roi_shape.size() - 1 - trailing_axis] : 1;
        OPENVINO_ASSERT(begin <= owner_shape[i] && dim <= owner_shape[i] - begin,
                        "ROI with offset ",
                        offset,
                        " and shape ",
                        roi_shape,
                        " exceeds tensor shape ",
                        owner_shape);
    }
    OPENVINO_ASSERT(remaining_offset == 0, "ROI offset must point to a tensor element.");
}

void copy_data(const std::shared_ptr<const ov::ITensor>& src,
               const std::shared_ptr<ov::ITensor>& dst,
               const ov::Shape& roi_shape,
               const uint8_t* src_data,
               uint8_t* dst_data) {
    if (ov::shape_size(roi_shape) == 0) {
        return;
    }
    const auto& type = src->get_element_type();
    if (type.bitwidth() < 8 ||
        (type != ov::element::string && src->get_shape() == roi_shape && dst->get_shape() == roi_shape &&
         src->get_strides() == dst->get_strides() && src->is_continuous())) {
        memcpy(dst_data, src_data, src->get_byte_size());
        return;
    }
    auto src_view = ov::make_tensor(type, roi_shape, src_data, src->get_strides_for_shape(roi_shape));
    auto dst_view = ov::make_tensor(type, roi_shape, dst_data, dst->get_strides_for_shape(roi_shape));
    src_view->copy_to(dst_view);
}

}  // namespace
namespace intel_npu {

ZeroRemoteTensor::ZeroRemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                   const ov::element::Type& element_type,
                                   const ov::Shape& shape,
                                   TensorType zero_tensor_type,
                                   MemType memory_type,
                                   const void* memory,
                                   const std::optional<FileDescriptor>& file_desc)
    : _context(context),
      _init_structs(init_structs),
      _element_type(element_type),
      _shape(shape),
      _capacity(shape),
      _logger("ZeroRemoteContext", Logger::global().level()),
      _tensor_type(zero_tensor_type),
      _mem_type(memory_type),
      _file_descriptor(file_desc),
      _mem(memory) {
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

    _host_memory = zero_mem::allocate_memory(_init_structs,
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
        _host_memory = zero_mem::allocate_memory(_init_structs,
                                                 bytes,
                                                 utils::STANDARD_PAGE_SIZE,
                                                 _tensor_type == TensorType::INPUT ? true : false);
        _data = _host_memory->data();
        break;
    }
    case MemType::SHARED_BUF: {
        // set up the request to import the external memory handle
        _host_memory = zero_mem::import_shared_memory(_init_structs, _mem, bytes);
        _data = _host_memory->data();
        break;
    }
    case MemType::MMAPED_FILE: {
        // File_descriptor shall be set if mem_type is a mmaped file type.
        OPENVINO_ASSERT(_file_descriptor.has_value(),
                        "No parameter ",
                        file_descriptor.name(),
                        " found in parameters map");

        if (_tensor_type == TensorType::OUTPUT) {
            // It is impossible to work on output today since ov::read_tensor_data opens the file in read-only mode.
            OPENVINO_THROW("Importing memory from a memory-mapped file is supported only for input tensors");
        } else if (_tensor_type == TensorType::BINDED) {
            _logger.warning("Importing memory from a memory-mapped file is supported only for input tensors");
            _tensor_type = TensorType::INPUT;
        }

        if (!_init_structs->isExternalMemoryStandardAllocationSupported()) {
            _logger.info("Importing mmaped memory isn't supported for this configuration. File data will be copied to "
                         "the level zero memory");
            copy_file_data_to_level_zero_memory(bytes);
            break;
        }

        // read-only tensor for mmaped file. So .data() should be called from a const context only.
        _mmap_tensor = ov::read_tensor_data(_file_descriptor.value()._file_path,
                                            _element_type,
                                            _shape,
                                            _file_descriptor.value()._offset_in_bytes,
                                            true);

        try {
            size_t aligned_size = utils::align_size_to_standard_page_size(bytes);
            _host_memory =
                zero_mem::import_standard_allocation_memory(_init_structs,
                                                            std::as_const(_mmap_tensor).data(),
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
    case MemType::CPU_VA: {
        _host_memory = zero_mem::import_standard_allocation_memory(_init_structs,
                                                                   _mem,
                                                                   bytes,
                                                                   _tensor_type == TensorType::INPUT ? true : false);
        _data = _host_memory->data();
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
    case MemType::CPU_VA:
        _properties = {mem_type(_mem_type), mem_handle(_data), tensor_type(_tensor_type)};
        break;
    case MemType::SHARED_BUF:
        _properties = {mem_type(_mem_type), mem_handle(_data), tensor_type(_tensor_type)};
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

void ZeroRemoteTensor::copy_to(const std::shared_ptr<ov::ITensor>& dst,
                               size_t src_offset,
                               size_t dst_offset,
                               const ov::Shape& roi_shape) const {
    OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
    OPENVINO_ASSERT(dst->get_element_type() == get_element_type(),
                    "Tensor element types are not equal. (src: ",
                    get_element_type(),
                    " != dst: ",
                    dst->get_element_type(),
                    ")");

    OPENVINO_ASSERT(roi_shape.empty() || get_element_type().bitwidth() >= 8,
                    "ROI copy is not supported for bitwidth < 8.");

    if (!roi_shape.empty()) {
        validate_roi_bounds(*this, src_offset, roi_shape);
        validate_roi_bounds(*dst, dst_offset, roi_shape);
    } else {
        if (get_shape() != dst->get_shape()) {
            dst->set_shape(get_shape());
        }
    }

    const auto& src_shape = roi_shape.empty() ? get_shape() : roi_shape;

    auto dst_zero_remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(dst);
    if (dst_zero_remote_tensor == nullptr) {
        if (auto remote_tensor_dst = std::dynamic_pointer_cast<ov::IRemoteTensor>(dst)) {
            OPENVINO_THROW("Copy to other remote tensor types is not supported.");
        }
    }

    auto* src_data = static_cast<const uint8_t*>(get_original_memory()) + src_offset;
    auto* dst_data = dst_zero_remote_tensor == nullptr
                         ? static_cast<uint8_t*>(dst->data()) + dst_offset
                         : static_cast<uint8_t*>(dst_zero_remote_tensor->get_original_memory()) + dst_offset;

    copy_data(shared_from_this(), dst, src_shape, src_data, dst_data);
}

void ZeroRemoteTensor::copy_from(const std::shared_ptr<const ov::ITensor>& src,
                                 size_t src_offset,
                                 size_t dst_offset,
                                 const ov::Shape& roi_shape) {
    OPENVINO_ASSERT(src, "Destination tensor was not initialized.");
    OPENVINO_ASSERT(src->get_element_type() == get_element_type(),
                    "Tensor element types are not equal. (src: ",
                    get_element_type(),
                    " != dst: ",
                    src->get_element_type(),
                    ")");

    OPENVINO_ASSERT(roi_shape.empty() || get_element_type().bitwidth() >= 8,
                    "ROI copy is not supported for bitwidth < 8.");

    if (!roi_shape.empty()) {
        validate_roi_bounds(*src, src_offset, roi_shape);
        validate_roi_bounds(*this, dst_offset, roi_shape);
    } else {
        if (src->get_shape() != get_shape()) {
            set_shape(src->get_shape());
        }
    }

    const auto& dst_shape = roi_shape.empty() ? get_shape() : roi_shape;

    auto src_zero_remote_tensor = std::dynamic_pointer_cast<const ZeroRemoteTensor>(src);
    if (src_zero_remote_tensor == nullptr) {
        if (auto remote_tensor_src = std::dynamic_pointer_cast<const ov::IRemoteTensor>(src)) {
            OPENVINO_THROW("Copy from other remote tensor types is not supported.");
        }
    }

    auto* src_data = src_zero_remote_tensor == nullptr
                         ? static_cast<const uint8_t*>(src->data()) + src_offset
                         : static_cast<const uint8_t*>(src_zero_remote_tensor->get_original_memory()) + src_offset;
    auto* dst_data = static_cast<uint8_t*>(get_original_memory()) + dst_offset;

    copy_data(src, shared_from_this(), dst_shape, src_data, dst_data);
}

ZeroRemoteTensor::~ZeroRemoteTensor() {
    _host_memory = nullptr;  // Ensure that zero memory is destroyed before the _mmap_tensor tensor is released
}

}  // namespace intel_npu
