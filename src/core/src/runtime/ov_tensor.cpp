// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "blob_factory.hpp"     // IE private header
#include "ie_ngraph_utils.hpp"  // IE private header
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/blob_allocator.hpp"
#include "shape_util.hpp"

namespace ov {

#define OV_TENSOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized."); \
    try {                                                             \
        __VA_ARGS__;                                                  \
    } catch (const std::exception& ex) {                              \
        throw ov::Exception(ex.what());                               \
    } catch (...) {                                                   \
        OPENVINO_ASSERT(false, "Unexpected exception");               \
    }

void Tensor::type_check(const Tensor&) {}

Tensor::~Tensor() {
    _impl = {};
}

Tensor::Tensor(const std::shared_ptr<ie::Blob>& impl, const std::vector<std::shared_ptr<void>>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, const Allocator& allocator) {
    OPENVINO_ASSERT(allocator, "Allocator was not initialized");
    auto allocator_impl = dynamic_cast<const BlobAllocator*>(allocator._impl.get());
    auto blob_allocator =
        (allocator_impl != nullptr) ? allocator_impl->_impl : std::make_shared<ie::BlobAllocator>(allocator._impl);
    _impl = make_blob_with_precision(
        {ie::details::convertPrecision(element_type), shape, ie::TensorDesc::getLayoutByDims(shape)},
        blob_allocator);
    _impl->allocate();
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, void* host_ptr, const Strides& byte_strides) {
    ie::SizeVector blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    ie::SizeVector dim_offset(shape.size(), 0);
    ie::SizeVector blk_strides;
    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else {
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " should be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }

    try {
        _impl = make_blob_with_precision(ie::details::convertPrecision(element_type),
                                         ie::TensorDesc{ie::details::convertPrecision(element_type),
                                                        shape,
                                                        ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}},
                                         host_ptr);
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end) : _so{owner._so} {
    OPENVINO_ASSERT(owner.get_element_type().bitwidth() >= 8,
                    "ROI Tensor for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                    owner.get_element_type());
    try {
        _impl = owner._impl->createROI(begin, end);
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

Tensor::Tensor(const ov::Output<const ov::Node>& port, const Allocator& allocator)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             allocator) {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& byte_strides)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             host_ptr,
             byte_strides) {}

element::Type Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return ie::details::convertPrecision(_impl->getTensorDesc().getPrecision()));
}

void Tensor::set_shape(const ov::Shape& shape) {
    // WA for tensor conversion from host tensor with dynamic shape.
    if (util::is_dynamic_shape(get_shape())) {
        _impl = make_blob_with_precision(
            {_impl->getTensorDesc().getPrecision(), shape, ie::TensorDesc::getLayoutByRank(shape.size())});
        _impl->allocate();
    } else {
        OV_TENSOR_STATEMENT(_impl->setShape({shape.begin(), shape.end()}));
    }
}

Shape Tensor::get_shape() const {
    OV_TENSOR_STATEMENT({ return _impl->getTensorDesc().getBlockingDesc().getBlockDims(); });
}

void Tensor::copy_to(ov::Tensor& dst) const {
    const auto& is_scalar = [](const ov::Shape& shape) {
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    };
    const auto shapes_equal = [is_scalar](const ov::Shape& src, const ov::Shape& dst) {
        // WA for scalar tensors to copy {1} to {} or otherwise
        return src == dst || (is_scalar(src) && is_scalar(dst));
    };
    OV_TENSOR_STATEMENT({
        OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
        OPENVINO_ASSERT(!is<ov::RemoteTensor>(), "Default copy to doesn't support copy from remote tensor.");
        OPENVINO_ASSERT(!dst.is<ov::RemoteTensor>(), "Default copy to doesn't support copy to remote tensor.");
        OPENVINO_ASSERT(dst.get_element_type() == get_element_type(),
                        "Tensor element types are not equal. (src: ",
                        get_element_type(),
                        " != dst: ",
                        dst.get_element_type(),
                        ")");
        if (dst.get_shape() == ov::Shape{0})
            dst.set_shape(get_shape());
        OPENVINO_ASSERT(shapes_equal(get_shape(), dst.get_shape()),
                        "Tensor shapes are not equal. (src: ",
                        get_shape(),
                        " != dst: ",
                        dst.get_shape(),
                        ")");
        const auto& shape = get_shape();
        auto* src_data = static_cast<const uint8_t*>(data());
        auto* dst_data = static_cast<uint8_t*>(dst.data());
        ov::Strides src_strides{get_byte_size()};
        ov::Strides dst_strides{dst.get_byte_size()};
        ov::Shape cur_pos{0};
        ov::Shape max_pos{1};

        if (get_element_type().bitwidth() < 8 || (get_strides() == dst.get_strides() && is_continuous()) ||
            (is_scalar(get_shape()) && is_scalar(dst.get_shape()))) {
            // OpenVINO doesn't support strides for LP types
            // or both tensors have default strides
            // Strides and positions already initialized
        } else {
            // Tensors have default strides
            const auto& type = get_element_type();
            std::vector<size_t> strides(shape.size());
            if (!shape.empty()) {
                strides[shape.size() - 1] = 1;
            }
            auto size = shape.size();
            for (size_t i = 1; i < size; i++) {
                strides[size - i - 1] = strides[size - i] * shape[size - i];
            }

            ov::Strides default_strides(strides.size());
            for (size_t i = 0; i < strides.size(); ++i)
                default_strides[i] = strides[i] * type.size();

            src_strides = get_strides();
            dst_strides = dst.get_strides();

            ov::Strides src_str, dst_str;

            // Calculate src and dst shapes
            bool found_step = false;
            for (size_t i = 0; i < shape.size(); i++) {
                size_t inverted_idx = shape.size() - i - 1;
                if (!found_step) {
                    if (default_strides[inverted_idx] == src_strides[inverted_idx] &&
                        src_strides[inverted_idx] == dst_strides[inverted_idx]) {
                        continue;
                    } else {
                        found_step = true;
                        size_t strides_size = inverted_idx + 1;
                        // Set right size
                        src_str.resize(strides_size + 1);
                        dst_str.resize(strides_size + 1);
                        max_pos.resize(strides_size + 1);
                        cur_pos.resize(strides_size + 1);
                        // In case of default continuous strides we can copy several elements
                        // In other case only one element
                        size_t dim = 1;
                        size_t strides = type.size();

                        if (strides_size < default_strides.size()) {
                            strides = default_strides[strides_size];
                            dim = get_shape()[strides_size];
                        }
                        src_str[strides_size] = strides;
                        dst_str[strides_size] = strides;
                        max_pos[strides_size] = dim;
                        cur_pos[strides_size] = 0;
                    }
                }
                src_str[inverted_idx] = src_strides[inverted_idx];
                dst_str[inverted_idx] = dst_strides[inverted_idx];
                max_pos[inverted_idx] = shape[inverted_idx];
                cur_pos[inverted_idx] = 0;
            }
            src_strides = src_str;
            dst_strides = dst_str;
        }

        const auto update_index = [](const ov::Shape& pos, const ov::Shape& shape, const ov::Strides& strides) {
            size_t offset = 0;

            for (size_t i = 0; i < pos.size(); i++) {
                offset += pos[i] * strides[i];
            }
            return offset;
        };

        bool finish = false;
        for (size_t dst_idx = 0, src_idx = 0; !finish;) {
            memcpy(dst_data + dst_idx, src_data + src_idx, src_strides[src_strides.size() - 1]);
            // update indexes
            for (size_t i = 0; i < cur_pos.size(); i++) {
                size_t inverted_idx = cur_pos.size() - i - 1;
                cur_pos[inverted_idx]++;
                if (cur_pos[inverted_idx] != max_pos[inverted_idx]) {
                    break;
                }
                if (inverted_idx)
                    cur_pos[inverted_idx] = 0;
                else
                    finish = true;
            }
            src_idx = update_index(cur_pos, max_pos, src_strides);
            dst_idx = update_index(cur_pos, max_pos, dst_strides);
        }
    });
}

Strides Tensor::get_strides() const {
    OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                    "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                    get_element_type());
    OV_TENSOR_STATEMENT({
        const auto& element_strides = _impl->getTensorDesc().getBlockingDesc().getStrides();
        const size_t elem_size = get_element_type().size();
        Strides byte_strides;
        byte_strides.resize(element_strides.size());
        std::transform(element_strides.begin(),
                       element_strides.end(),
                       byte_strides.begin(),
                       [&elem_size](size_t stride) {
                           return stride * elem_size;
                       });
        return byte_strides;
    });
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return _impl->size());
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return _impl->byteSize(););
}

void* Tensor::data(const element::Type element_type) const {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
#define TYPE_CHECK(TYPE) (dynamic_cast<const ie::TBlob<TYPE>*>(_impl.get()) != nullptr)
    auto host_accesable_implementation = TYPE_CHECK(bool) || TYPE_CHECK(int8_t) || TYPE_CHECK(uint8_t) ||
                                         TYPE_CHECK(int16_t) || TYPE_CHECK(uint16_t) || TYPE_CHECK(int32_t) ||
                                         TYPE_CHECK(uint32_t) || TYPE_CHECK(int64_t) || TYPE_CHECK(uint64_t) ||
                                         TYPE_CHECK(float) || TYPE_CHECK(double);
#undef TYPE_CHECK
    OPENVINO_ASSERT(host_accesable_implementation, "Tensor implementation type dose not contains host accessable data");
    if (element_type != element::undefined) {
        OPENVINO_ASSERT(element_type == get_element_type(),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
    }
    // since we don't use byte offsets, we need to explicitly multiply by element_size
    auto byte_offset = _impl->getTensorDesc().getBlockingDesc().getOffsetPadding() * get_element_type().size();
    OPENVINO_ASSERT((get_element_type().bitwidth() >= 8) || (byte_offset == 0),
                    "ROI access for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                    get_element_type());
    OV_TENSOR_STATEMENT(
        { return byte_offset + InferenceEngine::as<InferenceEngine::MemoryBlob>(_impl)->rmap().as<uint8_t*>(); });
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

bool Tensor::is_continuous() const {
    OV_TENSOR_STATEMENT({
        if (get_element_type().bitwidth() < 8)
            // OpenVINO doesn't support strides for lp types
            return true;
        const auto& shape = get_shape();
        const auto& type = get_element_type();
        std::vector<size_t> strides(shape.size());
        if (!shape.empty()) {
            strides[shape.size() - 1] = 1;
        }
        auto size = shape.size();
        for (size_t i = 1; i < size; i++) {
            strides[size - i - 1] = strides[size - i] * shape[size - i];
        }

        ov::Strides byte_strides(strides.size());
        for (size_t i = 0; i < strides.size(); ++i)
            byte_strides[i] = strides[i] * type.size();
        return byte_strides == get_strides();
    });
}

}  // namespace ov
