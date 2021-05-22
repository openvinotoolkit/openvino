// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for Blob and generic TBlob<>
 *
 * @file ie_blob.h
 */
#pragma once

#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ie_allocator.hpp"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_locked_memory.hpp"
#include "ie_precision.hpp"
#include "details/ie_blob_iterator.hpp"
#include "details/ie_pre_allocator.hpp"

namespace InferenceEngine {

/**
 * @brief This class represents a universal container in the Inference Engine
 *
 * @note Each Blob implementation must be derived from this Blob class directly or indirectly
 */
class INFERENCE_ENGINE_API_CLASS(Blob) {
public:
    /**
     * @brief A smart pointer containing Blob object
     */
    using Ptr = std::shared_ptr<Blob>;

    /**
     * @brief A smart pointer to the const Blob object
     */
    using CPtr = std::shared_ptr<const Blob>;

    /**
     * @brief Creates a TBlob<> object from a Data node
     *
     * @param data A reference to a smart pointer of the Data node
     * @return Smart pointer to TBlob<> with the relevant C type to the precision of the data node
     */
    static Ptr CreateFromData(const DataPtr& data);

    /**
     * @brief Blob virtual destructor
     */
    virtual ~Blob();

    /**
     * @brief Checks if the Blob object can be cast to the type T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the Blob
     * @return true if this object can be dynamically cast to the type T*. Otherwise, false
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    bool is() noexcept {
        return dynamic_cast<T*>(this) != nullptr;
    }

    /**
     * @brief Checks if the Blob object can be cast to the type const T*
     *
     * @tparam T Type to be checked. Must represent a class derived from the Blob
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    bool is() const noexcept {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    /**
     * @brief Casts this Blob object to the type T*.
     *
     * Use InferenceEngine::as() to operate with shared Blob objects instead of raw pointers
     *
     * @tparam T Type to cast to. Must represent a class derived from the Blob
     * @return Raw pointer to the object of the type T or nullptr on error
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    T* as() noexcept {
        return dynamic_cast<T*>(this);
    }

    /**
     * @brief Casts this Blob object to the type const T*.
     *
     * Use InferenceEngine::as() to operate with shared Blob objects instead of raw pointers
     *
     * @tparam T Type to cast to. Must represent a class derived from the Blob
     * @return Raw pointer to the object of the type const T or nullptr on error
     */
    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    const T* as() const noexcept {
        return dynamic_cast<const T*>(this);
    }

    /**
     * @brief Constructor. Creates an empty Blob object with the specified precision.
     *
     * @param tensorDesc Defines the layout and dims of the blob
     */
    explicit Blob(const TensorDesc& tensorDesc): tensorDesc(tensorDesc) {}

    /**
     * @brief Returns the tensor description
     * @return A const reference to a tensor descriptor
     */
    virtual const TensorDesc& getTensorDesc() const noexcept {
        return tensorDesc;
    }

    /**
     * @brief Returns the tensor description
     * @return A reference to a tensor descriptor
     */
    virtual TensorDesc& getTensorDesc() noexcept {
        return tensorDesc;
    }

    /**
     * @brief By default, returns the total number of elements (a product of all the dims or 1 for scalar)
     *
     * Return value and its interpretation heavily depend on the blob type
     *
     * @return The total number of elements
     */
    virtual size_t size() const noexcept {
        if (tensorDesc.getLayout() == Layout::SCALAR) return 1;
        return product(tensorDesc.getDims());
    }

    /**
     * @brief Returns the size of the current Blob in bytes.
     * @return Blob's size in bytes
     */
    virtual size_t byteSize() const noexcept {
        return size() * element_size();
    }

    /**
     * @deprecated Cast to MemoryBlob and use its API instead.
     * Blob class can represent compound blob, which do not refer to the only solid memory.
     *
     * @brief Provides the number of bytes per element.
     *
     * The overall Blob capacity is size() * element_size(). Abstract method.
     *
     * @return Returns the number of bytes per element
     */
    virtual size_t element_size() const noexcept = 0;

    /**
     * @brief Allocates memory to store the data.
     *
     * Abstract method.
     */
    virtual void allocate() noexcept = 0;

    /**
     * @brief Releases previously allocated data.
     *
     * Abstract method.
     *
     * @return `True` if deallocation happens successfully, `false` otherwise.
     */
    virtual bool deallocate() noexcept = 0;

    /**
     * @deprecated Cast to MemoryBlob and use new wlock/rwlock API instead.
     * Blob class can represent compound blob, which do not refer to the only solid memory.
     * @brief Gets access to the allocated memory.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    virtual LockedMemory<void> buffer() noexcept = 0;

    /**
     * @deprecated Cast to MemoryBlob and use new MemoryBlob::rmap() function instead.
     * Blob class can represent compound blob, which do not refer to the only solid memory.
     * @brief Gets read-only access to the allocated memory.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    virtual LockedMemory<const void> cbuffer() const noexcept = 0;

    /**
     * @brief Creates a blob describing given ROI object based on the current blob with memory sharing.
     *
     * Note: default implementation throws "not implemented" exception.
     *
     * @param roi A ROI object inside of the current blob.
     *
     * @return A shared pointer to the newly created ROI blob.
     */
    virtual Blob::Ptr createROI(const ROI& roi) const;

protected:
    /**
     * @brief The tensor descriptor of the given blob.
     */
    TensorDesc tensorDesc;

    /**
     * @deprecated Cast to MemoryBlob and use its API instead.
     * @brief Multiplies the dimension vector values.
     *
     * @param dims Reference to a vector with dimension values of type size_t
     * @return Result of multiplication
     */
    static size_t product(const SizeVector& dims) noexcept {
        if (dims.empty()) return 0;
        return std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
    }

    /**
     * @brief Gets an allocator for allocator-based blobs
     *
     * @return The allocator for allocator-based blobs or nullptr if there is none
     */
    virtual const std::shared_ptr<IAllocator>& getAllocator() const noexcept = 0;

    /**
     * @brief Gets a handle to allocated memory
     *
     * @return The handle to allocated memory for allocator-based blobs or nullptr if there is none
     */
    virtual void* getHandle() const noexcept = 0;

    /// private
    template <typename>
    friend class TBlobProxy;
};

/**
 * @brief Helper cast function to work with shared Blob objects
 * @param blob A blob to cast
 * @return shared_ptr to the type T. Returned shared_ptr shares ownership of the object with the
 *         input Blob::Ptr
 */
template <typename T,
          typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
          typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<T> as(const Blob::Ptr& blob) noexcept {
    return std::dynamic_pointer_cast<T>(blob);
}

/**
 * @brief Helper cast function to work with shared Blob objects
 * @param blob A blob to cast
 * @return shared_ptr to the type const T. Returned shared_ptr shares ownership of the object with
 *         the input Blob::Ptr
 */
template <typename T,
          typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
          typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<const T> as(const Blob::CPtr& blob) noexcept {
    return std::dynamic_pointer_cast<const T>(blob);
}

/**
 * @brief This class implements a container object that represents a tensor in memory (host and
 * remote/accelerated)
 *
 * @note Any Blob implementation that represents a concept of a tensor in memory (for example,
 * TBlob) must be a subclass of MemoryBlob instead of Blob
 */
class INFERENCE_ENGINE_API_CLASS(MemoryBlob): public Blob {
public:
    /**
     * @brief A smart pointer to the MemoryBlob object
     */
    using Ptr = std::shared_ptr<MemoryBlob>;

    /**
     * @brief A smart pointer to the const MemoryBlob object
     */
    using CPtr = std::shared_ptr<const MemoryBlob>;

    /**
     * @brief MemoryBlob virtual destructor
     */
    virtual ~MemoryBlob();

    /**
     * @brief Constructor. Creates an empty MemoryBlob object with the specified precision.
     *
     * @param tensorDesc Defines the layout and dims of the blob
     */
    explicit MemoryBlob(const TensorDesc& tensorDesc): Blob(tensorDesc) {}

    /**
     * @brief Returns the tensor description
     */
    const TensorDesc& getTensorDesc() const noexcept override {
        return tensorDesc;
    }

    /**
     * @brief Returns the tensor description
     */
    TensorDesc& getTensorDesc() noexcept override {
        return tensorDesc;
    }

    /**
     * @brief Returns the total number of elements, which is a product of all the dimensions
     * @return The total number of elements
     */
    size_t size() const noexcept override {
        if (tensorDesc.getLayout() == Layout::SCALAR) return 1;
        return product(tensorDesc.getDims());
    }

    /**
     * @brief Returns the size of the current Blob in bytes calculated as `size() * element_size()`.
     * @return Blob's size in bytes
     */
    size_t byteSize() const noexcept override {
        return size() * element_size();
    }

    /**
     * @brief Provides the number of bytes per element.
     * Abstract method.
     * @return The number of bytes per element.
     */
    size_t element_size() const noexcept override = 0;

    /**
     * @brief Allocates memory to store the data.
     *
     * Abstract method.
     */
    void allocate() noexcept override = 0;

    /**
     * @brief Releases previously allocated data.
     *
     * Abstract method.
     * @return `True` if deallocation happens successfully, `false` otherwise.
     */
    bool deallocate() noexcept override = 0;

    /**
     * @deprecated Use wmap() or rwmap() API instead.
     * @brief Gets access to the allocated memory.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    LockedMemory<void> buffer() noexcept override = 0;

    /**
     * @deprecated Use rmap() function instead.
     * @brief Gets read-only access to the allocated memory.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    LockedMemory<const void> cbuffer() const noexcept override = 0;

    /**
     * @brief Gets read/write access to the memory in virtual space of the process.
     * The function returns object which retains mapped memory.
     * The memory been addressed in the MemoryBlob in general case can be allocated on remote device.
     * This function maps remote memory to the memory in the virtual process space and after destruction
     * of the LockedMemory will upload changed content to the accelerator.
     *
     * To avoid extra copy of data, you can use rmap() and wmap() functions.
     *
     * In case of memory originally allocated on the host, this function returns LockedMemory which will
     * transparently refer to original memory address. No extra copy will happen
     *
     * In general case, pointer received from that LockedMemory becomes invalid just after
     * destruction of LockedMemory instance. Keep Locked memory alive while you need to address memory
     * in the process on the host.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    virtual LockedMemory<void> rwmap()noexcept = 0;

    /**
     * @brief Gets read only access to the memory in virtual space of the process.
     * The function returns object which retains mapped memory.
     *
     * The memory been addressed in the MemoryBlob in general case can be allocated on remote device.
     * This function copies remote memory to the memory in the virtual process space and after
     * destruction of the LockedMemory it will not upload host memory back, because it is expected that
     * content is not changed.
     *
     * To have an ability change content, you can use rwmap() and wmap() functions.
     *
     * In case of memory originally allocated on the host, this function returns LockedMemory which will
     * transparently refer to original memory address. No extra copy will happen
     *
     * In general case, pointer received from that LockedMemory becomes invalid just after destruction
     * of LockedMemory instance. Keep Locked memory alive while you need to address memory in the
     * process on the host.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    virtual LockedMemory<const void> rmap()const noexcept = 0;

    /**
     * @brief Gets "write only direction" access to the memory in virtual space of the process.
     * The function returns object which retains memory to be uploaded on device.
     *
     * The memory been addressed in the MemoryBlob in general case can be allocated on remote device.
     * This function does not copy of the content from the device to the memory in the virtual process
     * space, the content of the memory just after calling of this function is not specified. After
     * destruction of the LockedMemory, content will be upload host memory.
     * In the same time there is no abilities to restrict reading from the memory, you need to care of
     * reading from memory got by wmap(), it might have sense in some cases like filling of content and
     * before uploading to device
     *
     * To access data stored in the blob, you can use rwmap() and rmap() functions.
     *
     * In case of memory originally allocated on the host, this function returns LockedMemory which will
     * transparently refer to original memory address. No extra copy will happen
     *
     * In general case, pointer received from that LockedMemory becomes invalid just after destruction
     * of LockedMemory instance. Keep Locked memory alive while you need to address memory in the
     * process on the host.
     *
     * Abstract method.
     *
     * @return A LockedMemory object
     */
    virtual LockedMemory<void> wmap()noexcept = 0;

protected:
    /**
     * @brief Gets the allocator for allocator-based blobs.
     *
     * @return The allocator for allocator-based blobs or if there is none then a nullptr.
     */
    const std::shared_ptr<IAllocator>& getAllocator() const noexcept override = 0;

    /**
     * @brief Gets the handle to allocated memory.
     *
     * @return The handle to allocated memory for allocator-based blobs or if there is none then a nullptr.
     */
    void* getHandle() const noexcept override = 0;

    /// private
    template <typename>
    friend class TBlobProxy;
};

/**
 * @brief This is a convenient type for working with a map containing pairs(string, pointer to a Blob instance).
 */
using BlobMap = std::map<std::string, Blob::Ptr>;

/**
 * @brief Represents real host memory allocated for a Tensor/Blob per C type.
 */
template <typename T, typename = std::enable_if<std::is_pod<T>::value>>
class TBlob : public MemoryBlob {
    template <typename, typename>
    friend class TBlob;

public:
    /**
     * @brief Smart Pointer to this TBlob object.
     */
    using Ptr = std::shared_ptr<TBlob<T>>;

    /**
     * @brief Creates a TBlob object with the specified dimensions and layout but does not allocate the memory.
     *
     * Use the allocate() method to allocate memory.
     *
     * @param tensorDesc Tensor description
     */
    explicit TBlob(const TensorDesc& tensorDesc): MemoryBlob(tensorDesc) {}

    /**
     * @brief The constructor creates a TBlob object with the specified dimensions and layout
     * on the pre-allocated memory.
     *
     * The allocate() call is not required.
     *
     * @param tensorDesc Tensor description
     * @param ptr Pointer to the pre-allocated memory
     * @param data_size Length of the pre-allocated array. If not set, size is assumed equal
     * to the dot product of dims.
     */
    TBlob(const TensorDesc& tensorDesc, T* ptr, size_t data_size = 0): MemoryBlob(tensorDesc) {
        if (data_size == 0) {
            data_size = size();
        }

        if (data_size != 0 && ptr == nullptr) {
            IE_THROW() << "Using Blob on external nullptr memory";
        }

        _allocator = details::make_pre_allocator(ptr, data_size);
        // blob on attached memory is always allocated, so we are not forcing the user to call allocate()
        allocate();
    }

    /**
     * @brief Creates a TBlob object with the specified dimensions, layout and custom memory allocator but does not
     * allocate the memory.
     *
     * @param tensorDesc Tensor description
     * @param alloc An allocator
     */
    TBlob(const TensorDesc& tensorDesc, const std::shared_ptr<IAllocator>& alloc)
        : MemoryBlob(tensorDesc), _allocator(alloc) {
        if (_allocator == nullptr) IE_THROW() << "TBlob allocator was not initialized.";
    }

    /**
     * @brief The copy constructor data is reallocated and copied from the source to the target blob.
     *
     * @param blob Source blob
     */
    TBlob(const TBlob<T>& blob): MemoryBlob(blob.getTensorDesc()) {
        copyFrom(blob);
    }

    /**
     * @brief A move constructor.
     *
     * @param blob rvalue to make a move from
     */
    TBlob(TBlob<T>&& blob): MemoryBlob(blob.getTensorDesc()) {
        moveFrom(blob);
    }

    /**
     * @brief Copy operator for the TBlob object.
     *
     * @param blob object reference to copy from
     * @return Newly copied object
     */
    TBlob& operator=(const TBlob& blob) {
        copyFrom(blob);
        return *this;
    }

    /**
     *@brief Virtual destructor.
     */
    virtual ~TBlob();

    /**
     * @brief Gets the size of the given type.
     *
     * @return Size of the type
     */
    size_t element_size() const noexcept override {
        return sizeof(T);
    }

    /**
     * @brief Creates an new empty rvalue LockedMemory object.
     *
     * @return rvalue for the empty locked object of type T
     */
    virtual LockedMemory<T> data() noexcept {
        return std::move(lockme<T>());
    }

    /**
     * @brief Creates a new empty rvalue read-only LockedMemory object.
     *
     * @return rvalue for the empty locked const object of type T.
     */
    virtual LockedMemory<const T> readOnly() const noexcept {
        return std::move(lockme<const T>());
    }

    /**
     * @brief Allocates or reallocates memory
     */
    void allocate() noexcept override {
        const auto allocator = getAllocator();
        const auto rawHandle = allocator->alloc(size() * sizeof(T));

        if (rawHandle == nullptr) {
            return;
        }

        _handle.reset(
            rawHandle,
            [allocator](void* rawHandle) {
                allocator->free(rawHandle);
            });
    }

    /**
     * @brief Frees all allocated data
     */
    bool deallocate() noexcept override {
        return free();
    }

    /**
     * @brief Creates a new LockedMemory instance holding void pointer.
     *
     * @return LockedMemory instance holding void pointer
     */
    LockedMemory<void> buffer() noexcept override {
        return std::move(lockme<void>());
    }

    /**
     * @brief Creates a new LockedMemory instance holding constant void pointer.
     *
     * @return LockedMemory instance holding constant void pointer
     */
    LockedMemory<const void> cbuffer() const noexcept override {
        return std::move(lockme<const void>());
    }

    LockedMemory<void> rwmap()noexcept override {
        return std::move(lockme<void>());
    }

    LockedMemory<const void> rmap() const noexcept override {
        return std::move(lockme<const void>());
    }
    LockedMemory<void> wmap()noexcept override {
        return std::move(lockme<void>());
    }

    Blob::Ptr createROI(const ROI& roi) const override {
        return Blob::Ptr(new TBlob<T>(*this, roi));
    }

    /**
     * @brief Gets BlobIterator for the data.
     *
     * Enables a ranged loop support for the TBlob object.
     *
     * @return BlobIterator object of type T
     */
    details::BlobIterator<T> begin() {
        return details::BlobIterator<T>(data());
    }

    /**
     * @brief Gets BlobIterator for the end of data.
     *
     * Enables a ranged loop support for the TBlob object.
     *
     * @return BlobIterator object of type T representing end of the data
     */
    details::BlobIterator<T> end() {
        return details::BlobIterator<T>(data(), size());
    }

    /**
     * @brief Gets a const BlobIterator for the read-only data.
     *
     * Enables a ranged loop support for the TBlob object.
     *
     * @return BlobIterator object of type const T
     */
    details::BlobIterator<const T> begin() const {
        return details::BlobIterator<const T>(readOnly());
    }

    /**
     * @brief Gets a const BlobIterator for the end of read-only data.
     *
     * Enables a ranged loop support for the TBlob object.
     *
     * @return BlobIterator object of type const T representing end of data
     */
    details::BlobIterator<const T> end() const {
        return details::BlobIterator<const T>(readOnly(), size());
    }

protected:
    /**
     * @brief Local instance of IAllocator to manipulate memory.
     */
    mutable std::shared_ptr<IAllocator> _allocator;

    /**
     * @brief A handle for the stored memory returned from _allocator.alloc().
     */
    std::shared_ptr<void> _handle;

    /**
     * @brief Copies dimensions and data from the TBlob object.
     *
     * @param blob object reference to copy from
     */
    void copyFrom(const TBlob<T>& blob) {
        tensorDesc = blob.tensorDesc;
        this->allocate();
        auto memptr = data();
        memcpy(memptr, blob.readOnly(), byteSize());
    }

    /**
     * @brief Swaps memory handlers between the current blob and the given one.
     *
     * @tparam U Type of the blob to move from
     * @param blob TBlob instance to move from
     */
    template <class U>
    void moveFrom(TBlob<U>& blob) {
        tensorDesc = blob.tensorDesc;
        this->_allocator = std::move(blob._allocator);
        std::swap(this->_handle, blob._handle);
    }

    /**
     * @brief Frees handler and cleans up the stored data.
     */
    virtual bool free() {
        bool bCanRelease = _handle != nullptr;
        _handle.reset();
        return bCanRelease;
    }

    /**
     * @brief Creates a LockedMemory instance.
     *
     * @tparam S Type of the LockedMemory to be created
     * @return A created instance of LockedMemory
     */
    template <class S>
    LockedMemory<S> lockme() const {
        return LockedMemory<S>(_allocator.get(), getHandle(), 0);
    }

    /**
     * @brief Gets an allocator or creates a default one.
     *
     * @return IAllocator instance
     */
    const std::shared_ptr<IAllocator>& getAllocator() const noexcept override {
        // in case when constructor without allocator was used
        if (!_allocator) {
            _allocator = CreateDefaultAllocator();
        }

        return _allocator;
    }

    /**
     * @brief Returns handle to the stored data.
     */
    void* getHandle() const noexcept override {
        return _handle.get();
    }

    /**
     * @brief Creates a blob from the existing blob with a given ROI
     * @param origBlob An original blob
     * @param roi A ROI object
     */
    TBlob(const TBlob& origBlob, const ROI& roi) :
            MemoryBlob(make_roi_desc(origBlob.getTensorDesc(), roi, true)),
            _allocator(origBlob._allocator) {
        IE_ASSERT(origBlob._handle != nullptr)
            << "Original Blob must be allocated before ROI creation";

        _handle = origBlob._handle;
    }
};

#ifdef __clang__
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<float>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<double>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<int8_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<uint8_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<int16_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<uint16_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<int32_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<uint32_t>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<long>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<long long>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<unsigned long>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<unsigned long long>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<bool>);
extern template class INFERENCE_ENGINE_API_CLASS(InferenceEngine::TBlob<char>);
#endif  // __clang__

/**
 * @brief Creates a blob with the given tensor descriptor.
 *
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc Tensor descriptor for Blob creation
 * @return A shared pointer to the newly created blob of the given type
 */
template <typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(const TensorDesc& tensorDesc) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        IE_THROW() << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc);
}

/**
 * @brief Creates a blob with the given tensor descriptor from the pointer to the pre-allocated memory.
 *
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc TensorDesc for Blob creation
 * @param ptr Pointer to the pre-allocated memory
 * @param size Length of the pre-allocated array
 * @return A shared pointer to the newly created blob of the given type
 */
template <typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(const TensorDesc& tensorDesc, Type* ptr,
                                                                   size_t size = 0) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        IE_THROW() << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc, ptr, size);
}

/**
 * @brief Creates a blob with the given tensor descriptor and allocator.
 *
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc Tensor descriptor for Blob creation
 * @param alloc Shared pointer to IAllocator to use in the blob
 * @return A shared pointer to the newly created blob of the given type
 */
template <typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(
    const TensorDesc& tensorDesc, const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        IE_THROW() << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc, alloc);
}

/**
 * @brief Creates a copy of given TBlob instance.
 *
 * @tparam TypeTo Type of the shared pointer to be created
 * @param arg given pointer to blob
 * @return A shared pointer to the newly created blob of the given type
 */
template <typename TypeTo>
inline typename InferenceEngine::TBlob<TypeTo>::Ptr make_shared_blob(const TBlob<TypeTo>& arg) {
    return std::make_shared<InferenceEngine::TBlob<TypeTo>>(arg);
}

/**
 * @brief Creates a Blob object of the specified type
 *
 * @param args Constructor arguments for the Blob object
 * @return A shared pointer to the newly created Blob object
 */
template <typename T, typename... Args, typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<T> make_shared_blob(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

/**
 * @brief Creates a blob describing given ROI object based on the given blob with pre-allocated memory.
 *
 * @param inputBlob original blob with pre-allocated memory.
 * @param roi A ROI object inside of the original blob.
 * @return A shared pointer to the newly created blob.
 */
INFERENCE_ENGINE_API_CPP(Blob::Ptr) make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi);

}  // namespace InferenceEngine
