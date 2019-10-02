// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for Blob and generic TBlob<>
 * @file ie_blob.h
 */
#pragma once

#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <cstring>
#include <utility>
#include <functional>
#include <map>
#include <type_traits>

#include "ie_common.h"
#include "details/ie_exception.hpp"
#include "details/ie_blob_iterator.hpp"
#include "ie_allocator.hpp"
#include "ie_locked_memory.hpp"
#include "ie_precision.hpp"
#include "ie_layouts.h"
#include "details/ie_pre_allocator.hpp"

namespace InferenceEngine {
/**
 * @brief This class represents a universal container in the Inference Engine
 * @note Each Blob implementation must be derived from this Blob class directly or indirectly
 */
class Blob {
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
     * @param Data reference to a smart pointer of the Data node
     * @return Smart pointer to TBlob<> with the relevant C type to the precision of the data node
     */
    static Ptr CreateFromData(const DataPtr &data);

    /**
     * @brief Blob virtual destructor
     */
    virtual ~Blob()  = default;

    /**
     * @brief Checks if the Blob object can be cast to the type T*
     * @tparam T Type to be checked. Must represent a class derived from the Blob
     * @return true if this object can be dynamically cast to the type T*. Otherwise, false
     */
    template<typename T,
             typename std::enable_if<
                !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
             typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    bool is() noexcept {
        return dynamic_cast<T*>(this) != nullptr;
    }

    /**
     * @brief Checks if the Blob object can be cast to the type const T*
     * @tparam T Type to be checked. Must represent a class derived from the Blob
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template<typename T,
             typename std::enable_if<
                !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
             typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    bool is() const noexcept {
        return dynamic_cast<const T*>(this) != nullptr;
    }

    /**
     * @brief Casts this Blob object to the type T*. Use InferenceEngine::as() to operate with
     * shared Blob objects instead of raw pointers
     * @tparam T Type to cast to. Must represent a class derived from the Blob
     * @return Raw pointer to the object of the type T or nullptr on error
     */
    template<typename T,
             typename std::enable_if<
                !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
             typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    T* as() noexcept {
        return dynamic_cast<T*>(this);
    }

    /**
     * @brief Casts this Blob object to the type const T*. Use InferenceEngine::as() to operate with
     * shared Blob objects instead of raw pointers
     * @tparam T Type to cast to. Must represent a class derived from the Blob
     * @return Raw pointer to the object of the type const T or nullptr on error
     */
    template<typename T,
             typename std::enable_if<
                !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
             typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
    const T* as() const noexcept {
        return dynamic_cast<const T*>(this);
    }

    /**
     * @brief Constructor. Creates an empty Blob object with the specified precision.
     * @param tensorDesc Defines the layout and dims of the blob
     */
    explicit Blob(const TensorDesc &tensorDesc): tensorDesc(tensorDesc) {}

    /**
     * @brief Returns the tensor description
     */
    virtual const TensorDesc &getTensorDesc() const noexcept {
        return tensorDesc;
    }

    /**
     * @brief Returns the tensor description
     */
    virtual TensorDesc &getTensorDesc() noexcept {
        return tensorDesc;
    }

    /**
     * @brief By default, returns the total number of elements (a product of all the dims or 1 for scalar)
     *
     * Return value and its interpretation heavily depend on the blob type
     */
    virtual size_t size() const noexcept {
        if (tensorDesc.getLayout() == Layout::SCALAR)
            return 1;
        return product(tensorDesc.getDims());
    }

    /**
     * @brief Returns the size of the current Blob in bytes.
     */
    virtual size_t byteSize() const noexcept {
        return size() * element_size();
    }

    /**
     * @brief Returns the number of bytes per element. The overall Blob capacity is size() * element_size().
     * Abstract method.
     */
    virtual size_t element_size() const noexcept = 0;

    /**
     * @brief Allocates memory to store the data.
     * Abstract method.
     */
    virtual void allocate() noexcept = 0;

    /**
     * @brief Releases previously allocated data.
     * Abstract method.
     */
    virtual bool deallocate() noexcept = 0;

    /**
     * @brief Gets access to the allocated memory.
     * Abstract method.
     * @return A LockedMemory object
     */
    virtual LockedMemory<void> buffer() noexcept = 0;

    /**
     * @brief Gets read-only access to the allocated memory.
     * Abstract method.
     * @return A LockedMemory object
     */
    virtual LockedMemory<const void> cbuffer() const noexcept = 0;

protected:
    /**
     * @brief The tensor descriptor of the given blob.
     */
    TensorDesc tensorDesc;

    /**
     * @brief Multiplies the dimension vector's values.
     * @param dims Reference to a vector with dimension values of type size_t
     * @return Result of multiplication
     */
    static size_t product(const SizeVector &dims) noexcept {
        if (dims.empty())
            return 0;
        return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
    }

    /**
     * @brief Gets an allocator for allocator-based blobs
     * @return The allocator for allocator-based blobs or nullptr if there is none
     */
    virtual const std::shared_ptr<IAllocator> &getAllocator() const noexcept  = 0;

    /**
     * @brief Gets a handle to allocated memory
     * @return The handle to allocated memory for allocator-based blobs or nullptr if there is none
     */
    virtual void *getHandle() const noexcept  = 0;

    template<typename> friend
    class TBlobProxy;
};

/**
 * @brief Helper cast function to work with shared Blob objects
 * @return shared_ptr to the type T. Returned shared_ptr shares ownership of the object with the
 *         input Blob::Ptr
 */
template<typename T,
         typename std::enable_if<
            !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
         typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<T> as(const Blob::Ptr& blob) noexcept {
    return std::dynamic_pointer_cast<T>(blob);
}

/**
 * @brief Helper cast function to work with shared Blob objects
 * @return shared_ptr to the type const T. Returned shared_ptr shares ownership of the object with
 *         the input Blob::Ptr
 */
template<typename T,
         typename std::enable_if<
            !std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
         typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<const T> as(const Blob::CPtr& blob) noexcept {
    return std::dynamic_pointer_cast<const T>(blob);
}

/**
 * @brief This class implements a container object that represents a tensor in memory (host and
 * remote/accelerated)
 * @note Any Blob implementation that represents a concept of a tensor in memory (for example,
 * TBlob) must be a subclass of MemoryBlob instead of Blob
 */
class MemoryBlob : public Blob {
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
    virtual ~MemoryBlob()  = default;

    /**
     * @brief Constructor. Creates an empty MemoryBlob object with the specified precision.
     * @param tensorDesc Defines the layout and dims of the blob
     */
    explicit MemoryBlob(const TensorDesc& tensorDesc): Blob(tensorDesc) {}

    /**
     * @brief Returns the tensor description
     */
    const TensorDesc &getTensorDesc() const noexcept override {
        return tensorDesc;
    }

    /**
     * @brief Returns the tensor description
     */
    TensorDesc &getTensorDesc() noexcept override {
        return tensorDesc;
    }

    /**
     * @brief Returns the total number of elements, which is a product of all the dimensions
     */
    size_t size() const noexcept override {
        if (tensorDesc.getLayout() == Layout::SCALAR)
            return 1;
        return product(tensorDesc.getDims());
    }

    /**
     * @brief Returns the size of the current Blob in bytes
     */
    size_t byteSize() const noexcept override {
        return size() * element_size();
    }

    /**
     * @brief Returns the number of bytes per element. The overall MemoryBlob capacity is size() * element_size().
     * Abstract method.
     */
    size_t element_size() const noexcept override = 0;

    /**
     * @brief Allocates memory to store the data.
     * Abstract method.
     */
    void allocate() noexcept override = 0;

    /**
     * @brief Releases previously allocated data.
     * Abstract method.
     */
    bool deallocate() noexcept override = 0;

    /**
     * @brief Gets access to the allocated memory.
     * Abstract method.
     * @return A LockedMemory object
     */
    LockedMemory<void> buffer() noexcept override = 0;

    /**
     * @brief Gets read-only access to the allocated memory.
     * Abstract method.
     * @return A LockedMemory object
     */
    LockedMemory<const void> cbuffer() const noexcept override = 0;

protected:
    /**
    * @brief Gets the allocator for allocator-based blobs.
    * @return The allocator for allocator-based blobs or if there is none then a nullptr.
    */
    const std::shared_ptr<IAllocator> &getAllocator() const noexcept override  = 0;

    /**
    * @brief Gets the handle to allocated memory.
    * @return The handle to allocated memory for allocator-based blobs or if there is none then a nullptr.
    */
    void *getHandle() const noexcept override = 0;

    template<typename> friend
    class TBlobProxy;
};

/**
 * @brief This is a convenient type for working with a map containing pairs(string, pointer to a Blob instance).
 */
using BlobMap = std::map<std::string, Blob::Ptr>;

/**
 * @brief Represents real host memory allocated for a Tensor/Blob per C type.
 */
template<typename T,
        typename = std::enable_if<std::is_pod<T>::value>>
class TBlob : public MemoryBlob {
    template<typename, typename> friend
    class TBlob;


public:
    /**
     * @brief Smart Pointer to this TBlob object.
     */
    using  Ptr = std::shared_ptr<TBlob<T>>;

    /**
     * @brief Creates a TBlob object with the specified dimensions and layout but does not allocate the memory.
     * Use the allocate() method to allocate memory.
     * @param tensorDesc Tensor description
     */
    explicit TBlob(const TensorDesc& tensorDesc): MemoryBlob(tensorDesc) {}

    /**
     * @brief The constructor creates a TBlob object with the specified dimensions and layout
     * on the pre-allocated memory. The allocate() call is not required.
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
            THROW_IE_EXCEPTION << "Using Blob on external nullptr memory";
        }

        _allocator = details::make_pre_allocator(ptr, data_size);
        // blob on attached memory is always allocated, so we are not forcing the user to call allocate()
        allocate();
    }

    /**
     * @brief Creates a TBlob object with the specified dimensions, layout and custom memory allocator but does not allocate the memory.
     * @param p Precision
     * @param l Layout
     * @param dims Tensor dimensions
     * @param alloc Allocator to be used
     */
    TBlob(const TensorDesc& tensorDesc, const std::shared_ptr<IAllocator>& alloc)
            : MemoryBlob(tensorDesc), _allocator(alloc) {
    }

    /**
     * @brief The copy constructor data is reallocated and copied from the source to the target blob.
     * @param blob Source blob
     */
    TBlob(const TBlob<T> &blob) : MemoryBlob(blob.getTensorDesc()) {
        copyFrom(blob);
    }

    /**
     * @brief A move constructor.
     * @param blob rvalue to make a move from
     */
    TBlob(TBlob<T> &&blob) : MemoryBlob(blob.getTensorDesc()) {
        moveFrom(blob);
    }

    /**
     * @brief Copy operator for the TBlob object.
     * @param blob object reference to copy from
     * @return Newly copied object
     */
    TBlob &operator=(const TBlob &blob) {
        copyFrom(blob);
        return *this;
    }

    /**
     *@brief Virtual destructor.
     */
    virtual ~TBlob() {
        free();
    }

    /**
     * @brief Gets the size of the given type.
     * @return Size of the type
     */
    size_t element_size() const noexcept override {
        return sizeof(T);
    }

    /**
     * @brief Creates an new empty rvalue LockedMemory object.
     * @return rvalue for the empty locked object of type T
     */
    virtual LockedMemory<T> data() noexcept {
        return std::move(lockme<T>());
    }

    /**
     * @brief Creates a new empty rvalue read-only LockedMemory object.
     * @return rvalue for the empty locked const object of type T.
     */
    virtual LockedMemory<const T> readOnly() const noexcept {
        return std::move(lockme<const T>());
    }

    /**
     * @brief Allocates or reallocates memory
     */
    void allocate() noexcept override {
        if (_handle != nullptr) {
            getAllocator()->free(_handle);
        }
        _handle = getAllocator()->alloc(size() * sizeof(T));
    }

    /**
     * @brief Frees all allocated data
     */
    bool deallocate() noexcept override {
        return free();
    }

    /**
     * @brief Creates a new LockedMemory instance holding void pointer.
     * @return LockedMemory instance holding void pointer
     */
    LockedMemory<void> buffer() noexcept override {
        return std::move(lockme<void>());
    }

    /**
     * @brief Creates a new LockedMemory instance holding constant void pointer.
     * @return LockedMemory instance holding constant void pointer
     */
    LockedMemory<const void> cbuffer() const noexcept override {
        return std::move(lockme<const void>());
    }

    /**
     * @brief Gets BlobIterator for the data.
     * Enables a ranged loop support for the TBlob object.
     * @return BlobIterator object of type T
     */
    details::BlobIterator<T> begin() {
        return details::BlobIterator<T>(data());
    }

    /**
     * @brief Gets BlobIterator for the end of data.
     * Enables a ranged loop support for the TBlob object.
     * @return BlobIterator object of type T representing end of the data
     */
    details::BlobIterator<T> end() {
        return details::BlobIterator<T>(data(), size());
    }

    /**
     * @brief Gets a const BlobIterator for the read-only data.
     * Enables a ranged loop support for the TBlob object.
     * @return BlobIterator object of type const T
     */
    details::BlobIterator<const T> begin() const {
        return details::BlobIterator<const T>(readOnly());
    }

    /**
    * @brief Gets a const BlobIterator for the end of read-only data.
    * Enables a ranged loop support for the TBlob object.
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
    void *_handle = nullptr;

    /**
     * @brief Copies dimensions and data from the TBlob object.
     * @param blob object reference to copy from
     */
    void copyFrom(const TBlob<T> &blob) {
        tensorDesc = blob.tensorDesc;
        this->allocate();
        auto memptr = data();
        memcpy(memptr, blob.readOnly(), byteSize());
    }

    /**
     * @brief Swaps memory handlers between the current blob and the given one.
     * @tparam U Type of the blob to move from
     * @param blob TBlob instance to move from
     */
    template<class U>
    void moveFrom(TBlob<U> &blob) {
        tensorDesc = blob.tensorDesc;
        this->_allocator = std::move(blob._allocator);
        std::swap(this->_handle, blob._handle);
    }

    /**
     * @brief Frees handler and cleans up the stored data.
     */
    virtual bool free() {
        bool bCanRelease = getAllocator()->free(_handle);
        _handle = nullptr;
        return bCanRelease;
    }

    /**
     * @brief Creates a LockedMemory instance.
     * @tparam S Type of the LockedMemory to be created
     * @return A created instance of LockedMemory
     */
    template<class S>
    LockedMemory<S> lockme() const {
        return LockedMemory<S>(_allocator.get(), _handle, 0);
    }

    /**
     * @brief Gets an allocator or creates a default one.
     * @return IAllocator instance
     */
    const std::shared_ptr<IAllocator> &getAllocator() const noexcept override {
        // in case when constructor without allocator was used
        if (!_allocator) {
            _allocator = shared_from_irelease(CreateDefaultAllocator());
        }

        return _allocator;
    }

    /**
     * @brief Returns handle to the stored data.
     */
    void *getHandle() const noexcept override {
        return _handle;
    }
};

/**
 * @brief Creates a blob with the given tensor descriptor.
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc Tensor descriptor for Blob creation
 * @return A shared pointer to the newly created blob of the given type
 */
template<typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(const TensorDesc& tensorDesc) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        THROW_IE_EXCEPTION << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc);
}

/**
 * @brief Creates a blob with the given tensor descriptor from the pointer to the pre-allocated memory.
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc TensorDesc for Blob creation
 * @param ptr Pointer to the pre-allocated memory
 * @param size Length of the pre-allocated array
 * @return A shared pointer to the newly created blob of the given type
 */
template<typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(const TensorDesc& tensorDesc, Type * ptr, size_t size = 0) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        THROW_IE_EXCEPTION << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc, ptr, size);
}

/**
 * @brief Creates a blob with the given tensor descriptor and allocator.
 * @tparam Type Type of the shared pointer to be created
 * @param tensorDesc Tensor descriptor for Blob creation
 * @param alloc Shared pointer to IAllocator to use in the blob
 * @return A shared pointer to the newly created blob of the given type
 */
template<typename Type>
inline typename InferenceEngine::TBlob<Type>::Ptr make_shared_blob(const TensorDesc& tensorDesc, const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {
    if (!tensorDesc.getPrecision().hasStorageType<Type>())
        THROW_IE_EXCEPTION << "Cannot make shared blob! "
                           << "The blob type cannot be used to store objects of current precision";
    return std::make_shared<InferenceEngine::TBlob<Type>>(tensorDesc, alloc);
}

/**
 * @brief Creates a copy of given TBlob instance.
 * @tparam TypeTo Type of the shared pointer to be created
 * @param arg given pointer to blob
 * @return A shared pointer to the newly created blob of the given type
 */
template<typename TypeTo>
inline typename InferenceEngine::TBlob<TypeTo>::Ptr make_shared_blob(const TBlob<TypeTo> &arg) {
    return std::make_shared<InferenceEngine::TBlob<TypeTo>>(arg);
}

/**
 * @brief Creates a Blob object of the specified type
 * @param args Constructor arguments for the Blob object
 * @return A shared pointer to the newly created Blob object
 */
template<typename T, typename ...Args,
         typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type = 0>
std::shared_ptr<T> make_shared_blob(Args&& ...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

/**
 * @brief This structure describes ROI data.
 */
struct ROI {
    size_t id;     // ID of a ROI
    size_t posX;   // W upper left coordinate of ROI
    size_t posY;   // H upper left coordinate of ROI
    size_t sizeX;  // W size of ROI
    size_t sizeY;  // H size of ROI
};

/**
 * @brief Creates a blob describing given ROI object based on the given blob with pre-allocated memory.
 * @param inputBlob original blob with pre-allocated memory.
 * @param roi A ROI object inside of the original blob.
 * @return A shared pointer to the newly created blob.
 */
INFERENCE_ENGINE_API_CPP(Blob::Ptr) make_shared_blob(const Blob::Ptr &inputBlob, const ROI &roi);

}  // namespace InferenceEngine
