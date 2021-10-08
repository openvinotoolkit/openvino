// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layouts.h"
#include "memory_desc/cpu_memory_desc.h"
#include "mkldnn_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <mkldnn.hpp>
#include <mkldnn_types.h>
#include <cpu_shape.h>

#include "memory_desc/dnnl_memory_desc.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <ie_precision.hpp>

/**
 * @file contains a concept classes to work with memory/tensor/blob abstractions on plugin level.
 *
 * MKLDNNMemory is an abstraction of some real tensor which contains some data. As in short it's a pair of
 * memory descriptor and raw buffer handler to contains data. In case of system memory raw buffer it's simple
 * "void*" on some system memory buffer.
 *
 */

namespace MKLDNNPlugin {

class MKLDNNMemoryMngrInterface {
public:
    virtual ~MKLDNNMemoryMngrInterface() = default;
    virtual void* getRawPtr() const noexcept = 0;
    virtual void setExtBuff(void* ptr, size_t size) = 0;
    virtual void resize(size_t size) = 0;
    virtual bool hasExtBuffer() const noexcept = 0;
};

class MemoryMngrWithReuse : public MKLDNNMemoryMngrInterface {
public:
    MemoryMngrWithReuse() : data(nullptr, release) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    void resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

private:
    bool useExternalStorage = false;
    size_t memUpperBound = 0ul;
    std::unique_ptr<void, void (*)(void *)> data;

    static void release(void *ptr);
    static void destroy(void *ptr);
};

class MKLDNNMemory {
public:
    explicit MKLDNNMemory(const mkldnn::engine& eng);
    MKLDNNMemory(const mkldnn::engine& eng, std::unique_ptr<MKLDNNMemoryMngrInterface> mngr);

    MKLDNNMemory(const MKLDNNMemory&) = delete;
    MKLDNNMemory& operator= (const MKLDNNMemory&) = delete;

    MKLDNNMemory(MKLDNNMemory&&) = default;
    MKLDNNMemory& operator= (MKLDNNMemory&&) = default;

    mkldnn::memory GetPrimitive() const {
        if (isAllocated()) {
            return *prim;
        } else {
            IE_THROW() << "Can not perform GetPrimitive call to the not allocated memory";
        }
    }

    bool isAllocated() const noexcept {
        return prim != nullptr;
    }

    void setDataHandle(void* data);

    const MemoryDesc& getDesc() const {
        return *pMemDesc;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    std::shared_ptr<T> GetDescWithType() const;

    /**
     * Return handler of buffer. Real data may starts from some other offset
     * @return
     */
    void* GetData() const {
        void* data = pMngr->getRawPtr();
        if (data == nullptr &&
            pMemDesc->getShape().isStatic() &&
            pMemDesc->getShape().getElementsCount() != 0)
            IE_THROW() << "Cannot get memory!";
        return data;
    }

    /**
     * Return raw pointer on first element
     * Like a GetData() but offset is applied.
     * @return
     */
    void* GetPtr() const;

    mkldnn::memory::data_type GetDataType() const {
        return MKLDNNExtensionUtils::IEPrecisionToDataType(getDesc().getPrecision());
    }

    size_t GetSize() const;

    const Shape& GetShape() const {
        return getDesc().getShape();
    }

    void Create(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);
    void Create(MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);

    // Redefines descriptor. The memory descriptor will be replaced with the new one.
    // Memory will not be reallocated if the new tensor size is less or equal the upper bound.
    // Caution!!! This action invalidates the previous data layout. The old data may become unreachable.
    void redefineDesc(MemoryDescPtr desc);

    void SetData(const MKLDNNMemory& memory, size_t size = 0, bool ftz = true) const;
    void FillZero();

    const VectorDims& getStaticDims() const {
        return getDesc().getShape().getStaticDims();
    }

    mkldnn::engine getEngine() const {
        return eng;
    }

    bool isUsedExternalStorage() const {
        return pMngr->hasExtBuffer();
    }

private:
    void Create(const mkldnn::memory::dims& dims, mkldnn::memory::data_type data_type, mkldnn::memory::format_tag format,
                const void* data = nullptr);

    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr, bool pads_zeroing = true);

private:
    MemoryDescPtr pMemDesc;
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
    std::unique_ptr<MKLDNNMemoryMngrInterface> pMngr;
};

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;
using MKLDNNMemoryCPtr = std::shared_ptr<const MKLDNNMemory>;

}  // namespace MKLDNNPlugin
