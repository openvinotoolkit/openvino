// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/tensor.hpp"

#include <numeric>

#include "blob_allocator.hpp"
#include "blob_factory.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/except.hpp"

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

Tensor::Tensor(const std::shared_ptr<void>& so, const std::shared_ptr<ie::Blob>& impl) : _so{so}, _impl{impl} {
    if (_impl == nullptr) {
        IE_THROW() << "Tensor was not initialized.";
    }
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, const Allocator& allocator) {
    OPENVINO_ASSERT(allocator, "Allocator was not initalized");
    auto allocator_impl = dynamic_cast<const BlobAllocator*>(allocator._impl.get());
    auto blob_allocator =
        (allocator_impl != nullptr) ? allocator_impl->_impl : std::make_shared<ie::BlobAllocator>(allocator._impl);
    _impl = make_blob_with_precision({ie::details::convertPrecision(element_type),
                                      {shape.begin(), shape.end()},
                                      ie::TensorDesc::getLayoutByRank(shape.size())},
                                     blob_allocator);
    _impl->allocate();
}

Tensor::Tensor(const element::Type element_type,
               const Shape& shape,
               void* ptr,
               const size_t size,
               const Strides& strides) {
    ie::SizeVector blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    ie::SizeVector dim_offset(shape.size(), 0);
    ie::SizeVector blk_strides;
    if (strides.empty()) {
        blk_strides.assign(shape.begin(), shape.end());
    } else {
        blk_strides.assign(strides.begin(), strides.end());
    }
    _impl = make_blob_with_precision(ie::details::convertPrecision(element_type),
                                     ie::TensorDesc{ie::details::convertPrecision(element_type),
                                                    shape,
                                                    ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}},
                                     ptr,
                                     size);
}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end) : _so{owner._so} {
    try {
        _impl = owner._impl->createROI(begin, end);
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

element::Type Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return ie::details::convertPrecision(_impl->getTensorDesc().getPrecision()));
}

void Tensor::set_shape(const ov::Shape& shape) {
    OV_TENSOR_STATEMENT(_impl->setShape({shape.begin(), shape.end()}));
}

Shape Tensor::get_shape() const {
    OV_TENSOR_STATEMENT({
        auto dims = _impl->getTensorDesc().getDims();
        return {dims.begin(), dims.end()};
    });
}

Strides Tensor::get_strides() const {
    OV_TENSOR_STATEMENT(return _impl->getTensorDesc().getBlockingDesc().getStrides(););
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return ov::shape_size(get_shape()));
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return ov::shape_size(get_shape()) * get_element_type().size());
}

void* Tensor::data(const element::Type element_type) const {
    OV_TENSOR_STATEMENT({
        if (element_type != element::undefined) {
            OPENVINO_ASSERT(
                element::fundamental_type_for(element_type) == element::fundamental_type_for(get_element_type()),
                get_element_type(),
                " tensor fundamental element type is ",
                element::fundamental_type_for(get_element_type()),
                ", but it casted to ",
                element_type,
                " with fundamental element type",
                element::fundamental_type_for(element_type));
        }
        return _impl->getTensorDesc().getBlockingDesc().getOffsetPadding() * get_element_type().size() +
               InferenceEngine::as<InferenceEngine::MemoryBlob>(_impl)->rmap().as<uint8_t*>();
    });
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

}  // namespace ov
