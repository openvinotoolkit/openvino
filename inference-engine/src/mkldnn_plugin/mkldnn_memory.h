// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layouts.h"
#include "mkldnn_dims.h"
#include "cpu_memory_desc.h"
#include "mkldnn_extension_utils.h"
#include <mkldnn.hpp>
#include <mkldnn_types.h>
#include <cpu_shape.h>

#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <ie_precision.hpp>

/**
 * @file contains a concept classes to work with memory/tensor/blob abstractions on plugin level.
 *
 * MKLDNNMemoryDesc - the descriptor of tensor representation in memory. Describes all required information
 * for proper allocation and handling tensor in some buffer. The real memory is not present, just description.
 * This object answers on question how and where data with logical index [x1, x2, .. xN] placed in real buffer.
 * In the simplest case it describe a mapping between "logical offset" and "real offset".
 *
 * MKLDNNMemory is an abstraction of some real tensor which contains some data. As in short it's a pair of
 * memory descriptor and raw buffer handler to contains data. In case of system memory raw buffer it's simple
 * "void*" on some system memory buffer.
 *
 */

namespace MKLDNNPlugin {

/**
 * Represent internal plugin abstraction of tensor description
 *
 */
class MKLDNNMemoryDesc : public MemoryDesc {
public:
//    /** Empty constructor - doesn't define any tensor representation */
//    MKLDNNMemoryDesc() : MemoryDesc(Shape(), InferenceEngine::Precision::UNSPECIFIED, Mkldnn), desc() {}

    /** Construct a tensor desc with plain layout format (like ND C array) */
    MKLDNNMemoryDesc(const mkldnn::memory::dims& dims, mkldnn::memory::data_type dataType);

    /** Construct a tensor desc with specified layout format tag. Any and Undef is not supported */
    MKLDNNMemoryDesc(const mkldnn::memory::dims& dims, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format);

//    explicit MKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc);
    explicit MKLDNNMemoryDesc(const mkldnn::memory::desc& desc);

    /**
     * Try to define original format tag use on creation
     *
     * @return format tag if was able to define it
     */
    mkldnn::memory::format_tag getFormat() const;

    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    size_t GetElementSize() const;
    size_t getOffset(size_t elemNumber) const override;

    MKLDNNDims getDims() const {
        return MKLDNNDims(desc.data.dims, desc.data.ndims);
    }

    bool blocksExtended() const;
    operator bool() const {
        return getFormat() != mkldnn::memory::format_tag::any && getFormat() != mkldnn::memory::format_tag::undef;
    }

    bool operator == (const MKLDNNMemoryDesc& rhs) const;
    bool operator != (const MKLDNNMemoryDesc& rhs) const;

    operator mkldnn::memory::desc() const;

    bool isSame(mkldnn::memory::format_tag fmt) const;

    dnnl_format_kind_t getFormatKind() const {
        return desc.data.format_kind;
    }

    std::unique_ptr<MemoryDesc> clone() const override {
        return make_unique<MKLDNNMemoryDesc>(*this);
    }

    bool checkGeneralLayout(GeneralLayout layoutType) const override;

    std::string serializeFormat() const override;

    bool isDefined() const override;

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const BlockedMemoryDesc& rhs) const;
    bool isCompatible(const MKLDNNMemoryDesc& rhs) const;

private:
    size_t getMemSizeImp() const override;
    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size = UNREACHABLE_DIM) const;
    bool isTailCFormat() const;

private:
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();
    mkldnn::memory::desc desc;
};


class MKLDNNMemory {
public:
    explicit MKLDNNMemory(const mkldnn::engine& eng);

    MKLDNNMemory(const MKLDNNMemory&) = delete;
    MKLDNNMemory& operator= (const MKLDNNMemory&) = delete;

    MKLDNNMemory(MKLDNNMemory&&) = default;
    MKLDNNMemory& operator= (MKLDNNMemory&&) = default;

    const mkldnn::memory& GetPrimitive() const {
        return *prim;
    }

    const std::shared_ptr<mkldnn::memory>& GetPrimitivePtr() const {
        return prim;
    }

    mkldnn::memory::desc GetDescriptor() const {
        return prim->get_desc();
    }

    const MemoryDesc& GetDesc() const {
        return *pMemDesc;
    }

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    T GetDescWithType() const;

    /**
     * Return handler of buffer. Real data may starts from some other offset
     * @return
     */
    void* GetData() const {
        void* data = prim->get_data_handle();
        if (data == nullptr)
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
        return static_cast<mkldnn::memory::data_type>(GetDescriptor().data.data_type);
    }

    size_t GetSize() const;
    size_t GetElementsCount() const;

    mkldnn::memory::dims GetDims() const {
        auto data = GetDescriptor().data;
        return {std::begin(data.dims), std::begin(data.dims) + data.ndims};
    }

    void Create(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);

    // Like a plain format
    void SetData(mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format, const void* data, size_t size, bool ftz = true) const;
    void SetData(const MKLDNNMemory& memory, size_t size = 0, bool ftz = true) const;
    void FillZero();

    static mkldnn::memory::format_tag GetPlainFormatByRank(size_t rank);
    static InferenceEngine::Layout GetPlainLayout(const mkldnn::memory::dims& dims);
    static bool isConsistant(const mkldnn::memory::dims& dims, mkldnn::memory::format_tag format);
    static bool isConsistant(const Shape& dims, mkldnn::memory::format_tag format);
    static mkldnn::memory::format_tag Convert(const InferenceEngine::Layout layout);
    static InferenceEngine::Precision convertToIePrec(mkldnn::memory::data_type dataType);
    static mkldnn::memory::data_type convertToDataType(const InferenceEngine::Precision &precision);

    static std::string formatToString(mkldnn::memory::format_tag fmt);

    static void reorderData(const MKLDNNMemory& input, const MKLDNNMemory& output, size_t size = 0);

private:
    void Create(const mkldnn::memory::dims& dims, mkldnn::memory::data_type data_type, mkldnn::memory::format_tag format,
                const void* data = nullptr);

    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr, bool pads_zeroing = true);

    const MKLDNNMemoryDesc GetMKLDNNDesc() const {
        return MKLDNNMemoryDesc(prim->get_desc());
    }

private:
    MemoryDescPtr pMemDesc;
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
};

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;
using MKLDNNMemoryCPtr = std::shared_ptr<const MKLDNNMemory>;

}  // namespace MKLDNNPlugin
