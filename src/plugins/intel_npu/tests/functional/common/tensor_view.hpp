// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vpux/utils/core/error.hpp>

// Creates a wrapper around ov::Tensor and provides multi-dimensional checked element access
//
// Usage0: const auto tensor = ov::Tensor(ov::element::f32, ov::Shape{1, 24});
//         const auto tensorView = TensorView<float, 2>(tensor);
//         const float data = tensorView.at(n, c);
// Provided tensor must have number of dimensions equal to the template argument (checked at the runtime)
// Indices are bound-checked on each access
//
// Usage1: struct Box { float x0, y0, x1, y1; };
//         const auto tensor = ov::Tensor(ov::element::f32, ov::Shape{C, H, W}));
//         const auto boxesShape = ov::Shape{C, H, W / 4};
//         const auto boxesView = TensorView<Box, 3>(tensor, boxesShape);
//         const Box box = boxesView.at(c, h, w);  // 0 <= w < W / 4
// Interpret underlying data as a custom type.
// Inner dimension must be divisible by size of a template class (Box).
//
template <typename ElementType, int DimSize>
class TensorView {
private:
    TensorView(ElementType* data, ov::Shape shape): _data{data}, _shape{std::move(shape)} {
        VPUX_THROW_UNLESS(_data != nullptr, "Tensor data pointer is null");
    }

public:
    explicit TensorView(const ov::Tensor& tensor) {
        init(tensor);
    }

    explicit TensorView(const ov::Tensor& tensor, const ov::Shape& shape) {
        init(tensor, &shape);
    }

    // the secret to accessing private realms? self-friendship, of course
    template <typename U, int OtherDimSize>
    friend class TensorView;

    template <int NewDimSize>
    TensorView<ElementType, NewDimSize> reshape(ov::Shape shape) const {
        VPUX_THROW_UNLESS(shape.size() == NewDimSize, "Reshaping to {0}D tensor view, but {1}D shape provided",
                          NewDimSize, shape.size());
        VPUX_THROW_UNLESS(shape_size(this->_shape) == shape_size(shape),
                          "Failed to reshape tensor with shape {0} to shape {1}", this->_shape.to_string(),
                          shape.to_string());
        return TensorView<ElementType, NewDimSize>(_data, std::move(shape));
    }

    const ov::Shape& getShape() const {
        return _shape;
    }

    ElementType* data() const {
        return _data;
    }

    size_t size() const {
        return ov::shape_size(_shape);
    }

    template <typename... Indices, int D = DimSize, typename std::enable_if_t<(D == sizeof...(Indices)), int> = 0>
    ElementType& at(Indices&&... indices) {
        return getValue(std::forward<Indices>(indices)...);
    }

    template <typename... Indices, int D = DimSize, typename std::enable_if_t<(D == sizeof...(Indices)), int> = 0>
    ElementType& at(Indices&&... indices) const {
        return getValue(std::forward<Indices>(indices)...);
    }

private:
    void init(const ov::Tensor& tensor, const ov::Shape* shape = nullptr) {
        static_assert(!std::is_pointer_v<ElementType>, "Underlying type must not be a pointer");

        _data = reinterpret_cast<ElementType*>(tensor.data());
        _shape = tensor.get_shape();

        const auto typeSizeRatio = sizeof(ElementType) / tensor.get_element_type().size();
        VPUX_THROW_UNLESS(typeSizeRatio > 0, "Underlying type size={0} is greater than sizeof({1})={2}",
                          tensor.get_element_type().size(), typeid(ElementType).name(), sizeof(ElementType));

        VPUX_THROW_UNLESS(_shape.back() % typeSizeRatio == 0,
                          "Inner dimension of size={0} with sizeof({1})={2} is not divisible by sizeof({3})={4}",
                          _shape.back(), tensor.get_element_type().get_type_name(), tensor.get_element_type().size(),
                          typeid(ElementType).name(), sizeof(ElementType));
        _shape.back() /= typeSizeRatio;

        if (shape) {
            *this = this->reshape<DimSize>(*shape);
        }

        VPUX_THROW_UNLESS(_shape.size() == DimSize, "Creating {0}D tensor view for {1}D tensor", DimSize,
                          _shape.size());

        _strides.back() = 1;
        std::partial_sum(_shape.rbegin(), _shape.rend() - 1, _strides.rbegin() + 1, std::multiplies<>());
    }

    template <typename... Dims>
    ElementType& getValue(Dims... dims) const {
        static_assert(sizeof...(dims) == DimSize, "Invalid number of indices");

        auto dimsArray = std::array<int, sizeof...(Dims)>{dims...};
        int index = 0;

        for (auto i = static_cast<int>(dimsArray.size() - 1); i >= 0; i--) {
            VPUX_THROW_UNLESS((dimsArray[i] >= 0 && dimsArray[i] < _shape[i]),
                              "Index {0} with value {1} is out of bounds=[{2}..{3})", i, dimsArray[i], 0,
                              static_cast<int>(_shape[i]));
            index += dimsArray[i] * _strides[i];
        }

        return _data[index];
    }

private:
    ElementType* _data;
    ov::Shape _shape;
    std::array<int, DimSize> _strides;
};
