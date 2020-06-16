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
    MKLDNNMemoryDesc(): desc({}, mkldnn::memory::data_type::f32, mkldnn::memory::format::format_undef) {}
    explicit MKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc);
    explicit MKLDNNMemoryDesc(const mkldnn::memory::desc& desc): desc(desc) {}
    MKLDNNMemoryDesc(mkldnn::memory::dims dims, mkldnn::memory::data_type dataType, mkldnn::memory::format format);

    mkldnn::memory::format getFormat() const {
        return static_cast<mkldnn::memory::format>(desc.data.format);
    }

    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }
    ////////////////////////////////////////////
    /// TODO: Compatibility methods to simulate IE::TensorDesc API. Should be removed sometimes...
    InferenceEngine::Precision getPrecision() const;
    class BlkDesk {
        BlkDesk();
        mkldnn_blocking_desc_t _blk;
    public:
        bool operator == (const BlkDesk&& other) const;
    };
    BlkDesk getBlockingDesc() const;

    // TODO: it should removed. Just for build
    void setDataType(mkldnn::memory::data_type type) {
        desc.data.data_type = static_cast<mkldnn_data_type_t>(type);
        THROW_IE_EXCEPTION << "Should not be used";
    }
    ///////////

    MKLDNNDims getDims() const {
        return MKLDNNDims(desc.data.dims, desc.data.ndims);
    }

    bool blocksExtended() const;
    operator bool() const {
        return getFormat() != mkldnn::memory::format::any && getFormat() != mkldnn::memory::format::format_undef;
    }

    bool operator == (const MKLDNNMemoryDesc& rhs) const;
    bool operator != (const MKLDNNMemoryDesc& rhs) const;

    operator mkldnn::memory::desc() const;
    operator InferenceEngine::TensorDesc() const;

    bool isUnknown() const {
        return getFormat() == mkldnn::memory::format::any;
    }

    bool isDefined() const {
        return getFormat() != mkldnn::memory::format::any;
    }

    bool isUninit() const;

    MKLDNNMemoryDesc create_uninit_version() const;

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
        return prim->get_primitive_desc().desc();
    }

    mkldnn::memory::primitive_desc GetPrimitiveDescriptor() const {
        return prim->get_primitive_desc();
    }

    void* GetData() const {
        void* data = prim->get_data_handle();
        if (data == nullptr)
            THROW_IE_EXCEPTION << "Cannot get memory!";
        return data;
    }

    mkldnn::memory::data_type GetDataType() const {
        return static_cast<mkldnn::memory::data_type>(GetDescriptor().data.data_type);
    }

    size_t GetSize() const;
    size_t GetElementsCount() const;

    mkldnn::memory::format GetFormat() const {
        return static_cast<mkldnn::memory::format>(prim->get_primitive_desc().desc().data.format);
    }

    mkldnn::memory::dims GetDims() const {
        auto data = GetDescriptor().data;

        return std::vector<ptrdiff_t>(data.dims, data.dims + data.ndims);
    }

    void Create(mkldnn::memory::dims dims, mkldnn::memory::data_type data_type, mkldnn::memory::format format,
                const void* data = nullptr);

    void Create(const mkldnn::memory::desc& desc, const void* data = nullptr, bool pads_zeroing = true);

    void SetData(mkldnn::memory::data_type dataType, mkldnn::memory::format format, const void* data, size_t size, bool ftz = true) const;
    void SetData(const MKLDNNMemory& memory, bool ftz = true) const;

    void FillZero();

    static bool IsPlainFormat(mkldnn::memory::format format);
    static bool IsGroupedFormat(mkldnn::memory::format format);
    static mkldnn::memory::format GetPlainFormat(mkldnn::memory::dims dims);
    static InferenceEngine::Layout GetPlainLayout(mkldnn::memory::dims dims);
    static bool isConsistant(mkldnn::memory::dims dims, mkldnn::memory::format format);
    static mkldnn::memory::format Convert(const InferenceEngine::Layout layout);

    static std::string formatToString(mkldnn::memory::format fmt);

    static void CreateBlockingDesc(mkldnn::memory::desc& desc);

private:
    std::shared_ptr<mkldnn::memory> prim;
    mkldnn::engine eng;
};

using MKLDNNMemoryPtr = std::shared_ptr<MKLDNNMemory>;

/***********************************
 * Util section
 ***********************************/
mkldnn::memory::format get_format_tag(const InferenceEngine::TensorDesc &tdesc);
std::string format_tag_to_string(mkldnn::memory::format tag);

bool initTensorsAreEqual(const MKLDNNMemoryDesc left, const MKLDNNMemoryDesc right);

}  // namespace MKLDNNPlugin
