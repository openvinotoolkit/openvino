// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ie_layouts.h"
#include "mkldnn_dims.h"
#include <mkldnn.hpp>
#include <string>
#include <mkldnn_types.h>
#include <functional>

namespace MKLDNNPlugin {

class MKLDNNMemoryDesc {
public:
    MKLDNNMemoryDesc(): desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::undef) {}
    explicit MKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc);
    explicit MKLDNNMemoryDesc(const mkldnn::memory::desc& desc): desc(desc) {}
    MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format);

    mkldnn::memory::format_tag getFormat() const {
        // TODO: TBD
//        return static_cast<mkldnn::memory::format_tag>(desc.data.format);
        return mkldnn::memory::format_tag::undef;
    }

    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    size_t GetElementSize() const { THROW_IE_EXCEPTION << "Unimplemented"; };

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
    operator InferenceEngine::TensorDesc() const;

    bool isPlainFormat() const { THROW_IE_EXCEPTION << "UNIMPLEMENTED"; };
private:
    mkldnn::memory::desc desc;
};


class MKLDNNMemory {
public:
    explicit MKLDNNMemory(const mkldnn::engine& eng);

    const mkldnn::memory& GetPrimitive() const {
        return *prim;
    }

    const std::shared_ptr<mkldnn::memory>& GetPrimitivePtr() const {
        return prim;
    }

    mkldnn::memory::desc GetDescriptor() const {
        // TODO: TBD
//        return prim->get_primitive_desc().desc();
        return {};
    }

    const MKLDNNMemoryDesc GetDesc() const {
        // TODO: TBD
//        return prim->get_primitive_desc().desc();
        return {};
    }


    /// TODO: Should be avoid to use
//    mkldnn::memory::primitive_desc GetPrimitiveDescriptor() const {
//        return prim->get_primitive_desc();
//    }

    void* GetData() const {
        void* data = prim->get_data_handle();
        if (data == nullptr)
            THROW_IE_EXCEPTION << "Cannot get memory!";
        return data;
    }

    /**
     * Return raw pointer on first element
     * Like a GetData() but offset is applied.
     * @return
     */
    void* GetPtr() const {
        auto ptr = static_cast<uint8_t*>(GetData());
        ptr += GetDescriptor().data.offset0 * GetDesc().GetElementSize();
        return ptr;
    }


        mkldnn::memory::data_type GetDataType() const {
        return static_cast<mkldnn::memory::data_type>(GetDescriptor().data.data_type);
    }

    size_t GetSize() const;
    size_t GetElementsCount() const;


    mkldnn::memory::format_tag GetFormat() const {
        MKLDNNMemoryDesc desc(prim->get_desc());
        return desc.getFormat();
    }

    mkldnn::memory::dims GetDims() const {
        auto data = GetDescriptor().data;
        return {data.dims, data.dims + data.ndims};
    }

    void Create(mkldnn::memory::dims dims, mkldnn::memory::data_type data_type, mkldnn::memory::format_tag format,
                const void* data = nullptr);

    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr, bool pads_zeroing = true);

    void SetData(mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format, const void* data, size_t size, bool ftz = true) const;
    void SetData(const MKLDNNMemory& memory, bool ftz = true) const;
    void FillZero();

    bool IsPlain();

//    static bool IsPlainFormat(mkldnn::memory::format_tag format); // TODO: moved into instance method
//    static bool IsGroupedFormat(mkldnn::memory::format_tag format); // TODO: try to avoid usage
    static mkldnn::memory::format_tag GetPlainFormat(mkldnn::memory::dims dims);
    static InferenceEngine::Layout GetPlainLayout(mkldnn::memory::dims dims);
    static bool isConsistant(mkldnn::memory::dims dims, mkldnn::memory::format_tag format);
    static mkldnn::memory::format_tag Convert(const InferenceEngine::Layout layout);
    static InferenceEngine::Precision convertToIePrec(mkldnn::memory::data_type dataType);

    static std::string formatToString(mkldnn::memory::format_tag fmt);

    static void CreateBlockingDesc(mkldnn::memory::desc& desc);

private:
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
};

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;

}  // namespace MKLDNNPlugin
