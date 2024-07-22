// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shl_utils.hpp"
#include "csinn/csinn_data_structure.h"
#include "csinn/csinn_runtime.h"

#include "memory_desc/cpu_memory_desc.h"

#include <memory>


namespace ov {
namespace intel_cpu {


template <typename T>
struct ShlStructureTraits {};

template <typename T, typename traits = ShlStructureTraits<T>>
struct ShlStructure {
public:
    ShlStructure() = default;
    ShlStructure(const ShlStructure<T, traits>&) = default;
    ShlStructure(ShlStructure<T, traits>&&) = default;
    explicit ShlStructure(T t) { reset(t); }

    ShlStructure<T, traits> &operator=(const ShlStructure<T, traits>&) = default;
    ShlStructure<T, traits> &operator=(ShlStructure<T, traits>&&) = default;

    void reset(T t) {
        m_ptr.reset(t, traits::destructor);
    }

    T get(bool allow_empty = false) const {
        T result = m_ptr.get();
        OPENVINO_ASSERT(allow_empty || result != nullptr, "ShlStructure is not initialized");
        return result;
    }

    explicit operator T() const {
        return get(true);
    }

    explicit operator bool() const {
        return get(true) != nullptr;
    }

    bool operator==(const ShlStructure<T, traits> &other) const {
        return other.m_ptr.get() == m_ptr.get();
    }
    bool operator!=(const ShlStructure &other) const {
        return !(*this == other);
    }

private:
    std::shared_ptr<typename std::remove_pointer<T>::type> m_ptr = nullptr;

protected:
    bool operator==(const T other) const { return other == m_ptr.get(); }
    bool operator!=(const T other) const { return !(*this == other); }
};

template <>
struct ShlStructureTraits<csinn_session*> {
    static void destructor(csinn_session* p) {
        return csinn_free_session(p);
    }
};
struct ShlSession : public ShlStructure<csinn_session*> {
    ShlSession() {
        csinn_session* session = csinn_alloc_session();
        OPENVINO_ASSERT(session != nullptr, "Failed to create csinn_session");
        // CPU Plugin supports only per layer execution in SHL
        session->base_run_mode = CSINN_RM_LAYER;
        reset(session);
    }
};

template <>
struct ShlStructureTraits<csinn_tensor*> {
    static void destructor(csinn_tensor* p) {
        return csinn_free_tensor(p);
    }
};
struct ShlTensor : public ShlStructure<csinn_tensor*> {
    ShlTensor() {
        csinn_tensor* tensor = csinn_alloc_tensor(nullptr);
        OPENVINO_ASSERT(tensor != nullptr, "Failed to create csinn_tensor");
        reset(tensor);
    }

    ShlTensor(const ShlSession& session) {
        csinn_tensor* tensor = csinn_alloc_tensor(session.get());
        OPENVINO_ASSERT(tensor != nullptr, "Failed to create csinn_tensor");
        reset(tensor);
    }

    ShlTensor(const ShlSession& session, csinn_dtype_enum data_type, csinn_layout_enum layout, const VectorDims& shape = {}, void* data = nullptr)
        : ShlTensor(session) {
        setPrecision(data_type);
        setLayout(layout);
        setShape(shape);
        setData(data);
    }

    ShlTensor(const ShlTensor& another) : ShlTensor() {
        csinn_tensor_copy(get(), another.get());
    }

    csinn_layout_enum getLayout() const {
        // csinn_tensor contains `layout` as int32_t
        return static_cast<csinn_layout_enum>(get()->layout);
    }

    csinn_dtype_enum getPrecision() const {
        return get()->dtype;
    }

    VectorDims getShape() const {
        VectorDims shape(get()->dim_count);
        for (size_t i = 0; i < shape.size(); ++i)
            shape[i] = static_cast<size_t>(get()->dim[i]);
        return shape;
    }

    void* getData() const {
        return get()->data;
    }

    void setData(void* data) {
        get()->data = data;
    }

    ShlTensor cloneWithNewShape(const VectorDims& shape) const {
        ShlTensor cloned(*this);
        cloned.setShape(shape);
        return cloned;
    }

#ifdef CPU_DEBUG_CAPS
    void print() const {
        std::cout << "Shape: " << ov::Shape(getShape()) << " "
                  << "DataType: " << getPrecision() << " "
                  << "Layout: " << getLayout() << " "
                  << "Ptr: " << getData() << std::endl;
    }
#endif

private:
    void setLayout(csinn_layout_enum layout) {
        get()->layout = layout;
    }

    void setPrecision(csinn_dtype_enum data_type) {
        get()->dtype = data_type;
    }

    void setShape(const VectorDims& shape) {
        get()->dim_count = shape.size();
        OPENVINO_ASSERT(get()->dim_count < MAX_DIM, "Shl supports shapes with rank less or equal to 8");
        for (int i = 0; i < get()->dim_count; ++i)
            get()->dim[i] = static_cast<int32_t>(shape[i]);
    }
};

template <>
struct ShlStructureTraits<csinn_fc_params*> {
    static void destructor(csinn_fc_params* p) {
        return csinn_free_params(p);
    }
};
struct ShlFCParams : public ShlStructure<csinn_fc_params*> {
    ShlFCParams() {
        csinn_fc_params* params = static_cast<csinn_fc_params*>(csinn_alloc_params(sizeof(csinn_fc_params), nullptr));
        OPENVINO_ASSERT(params != nullptr, "Failed to create csinn_fc_params");
        reset(params);
    }

    ShlFCParams(const ShlSession& session, csinn_api_enum api) {
        csinn_fc_params* params = static_cast<csinn_fc_params*>(csinn_alloc_params(sizeof(csinn_fc_params), session.get()));
        OPENVINO_ASSERT(params != nullptr, "Failed to create csinn_fc_params");
        params->base.api = api;
        reset(params);
    }
};

}   // namespace intel_cpu
}   // namespace ov
