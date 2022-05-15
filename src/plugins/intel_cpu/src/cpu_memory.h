// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layouts.h"
#include "memory_desc/cpu_memory_desc.h"
#include "extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <mkldnn.hpp>
#include <mkldnn_types.h>
#include <cpu_shape.h>

#include "memory_desc/dnnl_memory_desc.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <mutex>
#include <ie_precision.hpp>

/**
 * @file contains a concept classes to work with memory/tensor/blob abstractions on plugin level.
 *
 * MKLDNNMemory is an abstraction of some real tensor which contains some data. As in short it's a pair of
 * memory descriptor and raw buffer handler to contains data. In case of system memory raw buffer it's simple
 * "void*" on some system memory buffer.
 *
 */

namespace ov {
namespace intel_cpu {

class MKLDNNMemory;

/**
 * @interface IMemoryMngr
 * @brief An interface to memory control object
 */

class IMemoryMngr {
public:
    virtual ~IMemoryMngr() = default;

    /**
     * @brief Accessor to underlying memory buffer
     * @return A pointer to underlying memory
     */
    virtual void* getRawPtr() const noexcept = 0;

    /**
     * @brief Allows to set externally allocated memory buffer. In that case, the object has no control over the provided memory.
     * @param ptr - pointer to the memory
     * @param size - size of the memory buffer
     */
    virtual void setExtBuff(void* ptr, size_t size) = 0;

    /**
     * @brief Resize underlying memory buffer
     * @param size - new memory size in bytes
     * @return status whether the memory reallocation was performed
     */
    virtual bool resize(size_t size) = 0;

    /**
     * @brief Check if the object has control over underlying memory buffer
     * @return status whether the object has control over underlying memory buffer
     */
    virtual bool hasExtBuffer() const noexcept = 0;
};

/**
 * @brief An implementation of the mem manager where memory reallocation occures only if bigger buffer is requested.
 */
class MemoryMngrWithReuse : public IMemoryMngr {
public:
    MemoryMngrWithReuse() : _data(nullptr, release) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

private:
    bool _useExternalStorage = false;
    size_t _memUpperBound = 0ul;
    std::unique_ptr<void, void (*)(void *)> _data;

    static void release(void *ptr);
    static void destroy(void *ptr);
};

/**
 * @brief A proxy object that additionally implements observer pattern
 */
class DnnlMemoryMngr : public IMemoryMngr {
public:
    explicit DnnlMemoryMngr(std::unique_ptr<IMemoryMngr> mngr) : _pMemMngr(std::move(mngr)) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void registerMemory(MKLDNNMemory* memPtr);
    void unregisterMemory(MKLDNNMemory* memPtr);

private:
    void notifyUpdate();

private:
    std::unordered_set<MKLDNNMemory*> _setMemPtrs;
    std::unique_ptr<IMemoryMngr> _pMemMngr;
};

using DnnlMemoryMngrPtr = std::shared_ptr<DnnlMemoryMngr>;
using DnnlMemoryMngrCPtr = std::shared_ptr<const DnnlMemoryMngr>;

class DnnlMemMngrHandle {
public:
    DnnlMemMngrHandle(DnnlMemoryMngrPtr pMgr, MKLDNNMemory* pMem) : _pMgr(pMgr), _pMem(pMem) {
        if (_pMgr) {
            _pMgr->registerMemory(_pMem);
        }
    }

    DnnlMemMngrHandle(const DnnlMemMngrHandle&) = delete;
    DnnlMemMngrHandle& operator= (const DnnlMemMngrHandle&) = delete;

    DnnlMemMngrHandle(DnnlMemMngrHandle&& source) {
        std::swap(_pMgr, source._pMgr);
        std::swap(_pMem, source._pMem);
    }
    DnnlMemMngrHandle& operator= (DnnlMemMngrHandle&& rhs) {
        std::swap(_pMgr, rhs._pMgr);
        std::swap(_pMem, rhs._pMem);
        return *this;
    }

    ~DnnlMemMngrHandle() {
        if (_pMgr) {
            _pMgr->unregisterMemory(_pMem);
        }
    }

    DnnlMemoryMngrPtr get() const {
        return _pMgr;
    }

    DnnlMemoryMngrPtr::element_type* operator->() const noexcept {
        return _pMgr.get();
    }

private:
    DnnlMemoryMngrPtr _pMgr = nullptr;
    MKLDNNMemory* _pMem = nullptr;
};

class MKLDNNMemory {
public:
    explicit MKLDNNMemory(const mkldnn::engine& eng);
    MKLDNNMemory(const mkldnn::engine& eng, std::unique_ptr<IMemoryMngr> mngr);

    MKLDNNMemory(const MKLDNNMemory&) = delete;
    MKLDNNMemory& operator= (const MKLDNNMemory&) = delete;

    MKLDNNMemory(MKLDNNMemory&&) = delete;
    MKLDNNMemory& operator= (MKLDNNMemory&&) = delete;

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

    /**
     * @brief Resets the memory manager to a new one created with the provided raw memory
     */
    void setDataHandle(void* data);

    const MemoryDesc& getDesc() const {
        return *pMemDesc;
    }

    MemoryDescPtr getDescPtr() const {
        return pMemDesc;
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
        std::lock_guard<std::mutex> lock(mutex);
        void* data = mgrHandle->getRawPtr();
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

    void Create(const MemoryDesc& desc, DnnlMemoryMngrPtr memMgr);
    void Create(MemoryDescPtr desc, DnnlMemoryMngrPtr memMgr);

    // Redefines descriptor. The memory descriptor will be replaced with the new one.
    // Memory will not be reallocated if the new tensor size is less or equal the upper bound.
    // Caution!!! This action invalidates the previous data layout. The old data may become unreachable.
    void redefineDesc(MemoryDescPtr desc);

    void SetData(const MKLDNNMemory& memory, bool ftz = true) const;
    void FillZero();

    const VectorDims& getStaticDims() const {
        return getDesc().getShape().getStaticDims();
    }

    mkldnn::engine getEngine() const {
        return eng;
    }

    bool isUsedExternalStorage() const {
        return mgrHandle->hasExtBuffer();
    }

    DnnlMemoryMngrPtr getDnnlMemoryMngr() const {
        return mgrHandle.get();
    }

private:
    friend DnnlMemoryMngr;

private:
    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr, bool pads_zeroing = true);
    void update();

private:
    MemoryDescPtr pMemDesc;
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
    DnnlMemMngrHandle mgrHandle;
    mutable std::mutex mutex;
};

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;
using MKLDNNMemoryCPtr = std::shared_ptr<const MKLDNNMemory>;

}   // namespace intel_cpu
}   // namespace ov
