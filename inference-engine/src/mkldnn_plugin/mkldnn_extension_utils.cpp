// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_extension_utils.h"
#include <limits>
#include <vector>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::MKLDNNPlugin;

uint8_t MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type dataType) {
    switch (dataType) {
    case mkldnn::memory::data_type::f32:
        return 4;
    case mkldnn::memory::data_type::s32:
        return 4;
    case mkldnn::memory::data_type::s16:
        return 2;
    case mkldnn::memory::data_type::s8:
        return 1;
    case mkldnn::memory::data_type::u8:
        return 1;
    default:
        THROW_IE_EXCEPTION << "Unsupported data type.";
    }
}

memory::data_type MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision prec) {
    switch (prec) {
        case InferenceEngine::Precision::FP32:
            return memory::f32;
        case InferenceEngine::Precision::I16:
            return memory::s16;
        case InferenceEngine::Precision::I8:
            return memory::s8;
        case InferenceEngine::Precision::U8:
            return memory::u8;

        default: {
            THROW_IE_EXCEPTION << "The plugin does not support " << prec.name();
        }
    }
}

InferenceEngine::Precision MKLDNNExtensionUtils::DataTypeToIEPrecision(memory::data_type dataType) {
    switch (dataType) {
        case memory::f32:
            return InferenceEngine::Precision(InferenceEngine::Precision::FP32);

        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

InferenceEngine::SizeVector MKLDNNExtensionUtils::MKLDimsToSizeVector(memory::dims dims) {
    InferenceEngine::SizeVector size;

    for (int i = 0; i < dims.size(); i++) {
        size.push_back(dims[i]);
    }

    return size;
}

MemoryFormat MKLDNNExtensionUtils::MKLFormatToMemoryFormat(memory::dims dims, memory::format fmt) {
    switch (fmt) {
        case memory::format_undef: return MemoryFormat::format_undef;
        case memory::any: return MemoryFormat::any;
        case memory::blocked: return MemoryFormat::blocked;
        case memory::x: return MemoryFormat::x;
        case memory::nc: return MemoryFormat::nc;
        case memory::nchw: return MemoryFormat::nchw;
        case memory::nhwc: return MemoryFormat::nhwc;
        case memory::chwn: return MemoryFormat::chwn;
        case memory::nChw8c: return MemoryFormat::nChw8c;
        case memory::nChw16c: return MemoryFormat::nChw16c;
        case memory::oi: return MemoryFormat::oi;
        case memory::io: return MemoryFormat::io;
        case memory::oihw: return MemoryFormat::oihw;
        case memory::ihwo: return MemoryFormat::ihwo;
        case memory::hwio: return MemoryFormat::hwio;
        case memory::OIhw8i8o: return MemoryFormat::OIhw8i8o;
        case memory::OIhw16i16o: return MemoryFormat::OIhw16i16o;
        case memory::OIhw8i16o2i: return MemoryFormat::OIhw8i16o2i;
        case memory::OIhw8o16i2o: return MemoryFormat::OIhw8o16i2o;
        case memory::OIhw8o8i: return MemoryFormat::OIhw8o8i;
        case memory::OIhw16o16i: return MemoryFormat::OIhw16o16i;
        case memory::Oihw8o: return MemoryFormat::Oihw8o;
        case memory::Oihw16o: return MemoryFormat::Oihw16o;
        case memory::Ohwi8o: return MemoryFormat::Ohwi8o;
        case memory::Ohwi16o: return MemoryFormat::Ohwi16o;
        case memory::OhIw16o4i: return MemoryFormat::OhIw16o4i;
        case memory::goihw: return MemoryFormat::goihw;
        case memory::gOIhw8i8o: return MemoryFormat::gOIhw8i8o;
        case memory::gOIhw16i16o: return MemoryFormat::gOIhw16i16o;
        case memory::gOIhw8i16o2i: return MemoryFormat::gOIhw8i16o2i;
        case memory::gOIhw8o16i2o: return MemoryFormat::gOIhw8o16i2o;
        case memory::gOIhw8o8i: return MemoryFormat::gOIhw8o8i;
        case memory::gOIhw16o16i: return MemoryFormat::gOIhw16o16i;
        case memory::gOihw8o: return MemoryFormat::gOihw8o;
        case memory::gOihw16o: return MemoryFormat::gOihw16o;
        case memory::gOhwi8o: return MemoryFormat::gOhwi8o;
        case memory::gOhwi16o: return MemoryFormat::gOhwi16o;
        case memory::gOhIw16o4i: return MemoryFormat::gOhIw16o4i;
        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

memory::format MKLDNNExtensionUtils::MemoryFormatToMKLFormat(MemoryFormat fmt) {
    switch (fmt) {
        case MemoryFormat::format_undef: return memory::format_undef;
        case MemoryFormat::any: return memory::any;
        case MemoryFormat::blocked: return memory::blocked;
        case MemoryFormat::x: return memory::x;
        case MemoryFormat::nc: return memory::nc;
        case MemoryFormat::nchw: return memory::nchw;
        case MemoryFormat::nhwc: return memory::nhwc;
        case MemoryFormat::chwn: return memory::chwn;
        case MemoryFormat::nChw8c: return memory::nChw8c;
        case MemoryFormat::nChw16c: return memory::nChw16c;
        case MemoryFormat::oi: return memory::oi;
        case MemoryFormat::io: return memory::io;
        case MemoryFormat::oihw: return memory::oihw;
        case MemoryFormat::ihwo: return memory::ihwo;
        case MemoryFormat::hwio: return memory::hwio;
        case MemoryFormat::OIhw8i8o: return memory::OIhw8i8o;
        case MemoryFormat::OIhw16i16o: return memory::OIhw16i16o;
        case MemoryFormat::OIhw8i16o2i: return memory::OIhw8i16o2i;
        case MemoryFormat::OIhw8o16i2o: return memory::OIhw8o16i2o;
        case MemoryFormat::OIhw8o8i: return memory::OIhw8o8i;
        case MemoryFormat::OIhw16o16i: return memory::OIhw16o16i;
        case MemoryFormat::Oihw8o: return memory::Oihw8o;
        case MemoryFormat::Oihw16o: return memory::Oihw16o;
        case MemoryFormat::Ohwi8o: return memory::Ohwi8o;
        case MemoryFormat::Ohwi16o: return memory::Ohwi16o;
        case MemoryFormat::OhIw16o4i: return memory::OhIw16o4i;
        case MemoryFormat::goihw: return memory::goihw;
        case MemoryFormat::gOIhw8i8o: return memory::gOIhw8i8o;
        case MemoryFormat::gOIhw16i16o: return memory::gOIhw16i16o;
        case MemoryFormat::gOIhw8i16o2i: return memory::gOIhw8i16o2i;
        case MemoryFormat::gOIhw8o16i2o: return memory::gOIhw8o16i2o;
        case MemoryFormat::gOIhw8o8i: return memory::gOIhw8o8i;
        case MemoryFormat::gOIhw16o16i: return memory::gOIhw16o16i;
        case MemoryFormat::gOihw8o: return memory::gOihw8o;
        case MemoryFormat::gOihw16o: return memory::gOihw16o;
        case MemoryFormat::gOhwi8o: return memory::gOhwi8o;
        case MemoryFormat::gOhwi16o: return memory::gOhwi16o;
        case MemoryFormat::gOhIw16o4i: return memory::gOhIw16o4i;
        default: {
            THROW_IE_EXCEPTION << "Unsupported data type.";
        }
    }
}

MKLDNNPrimitiveMemory MKLDNNExtensionUtils::MKLMemoryToGenericMemory(const MKLDNNMemory& mem) {
    MKLDNNPrimitiveMemory memory;

    memory.dims = MKLDimsToSizeVector(mem.GetDims());
    memory.data = mem.GetData();
    memory.precision = DataTypeToIEPrecision(mem.GetDataType());
    memory.format = MKLFormatToMemoryFormat(mem.GetDims(), mem.GetFormat());

    return memory;
}

InferenceEngine::TensorDesc MKLDNNExtensionUtils::getUninitTensorDesc(const InferenceEngine::TensorDesc &desc) {
    std::vector<size_t> notInitArr;
    std::vector<size_t> zeroArr;
    for (size_t i = 0; i < desc.getBlockingDesc().getBlockDims().size(); i++) {
        notInitArr.push_back(std::numeric_limits<size_t>::max());
        zeroArr.push_back(0);
    }
    // MKLDNN doesn't support offset_padding_to_data[i] != 0 (assert(src_d_blk.offset_padding_to_data[d] == 0);)
    return desc.getLayout() == InferenceEngine::Layout::ANY ? desc :
           InferenceEngine::TensorDesc(desc.getPrecision(), desc.getDims(),
                                       {desc.getBlockingDesc().getBlockDims(), desc.getBlockingDesc().getOrder(),
                                        std::numeric_limits<size_t>::max(), zeroArr, notInitArr});
}

bool MKLDNNExtensionUtils::initTensorsAreEqual(InferenceEngine::TensorDesc desc1, InferenceEngine::TensorDesc desc2) {
    if (desc1.getDims() != desc2.getDims() || desc1.getPrecision() != desc2.getPrecision())
        return false;
    if (desc1.getLayout() == InferenceEngine::Layout::ANY || desc2.getLayout() == InferenceEngine::Layout::ANY)
        return true;
    bool batch1 = desc1.getDims()[0] == 1;
    const auto& in1Block = desc1.getBlockingDesc();
    const auto& in2Block = desc2.getBlockingDesc();
    size_t uninitNum = std::numeric_limits<size_t>::max();
    if (in1Block.getBlockDims().size() != in2Block.getBlockDims().size())
        return false;
    for (size_t i = 0; i < in1Block.getBlockDims().size(); i++) {
        if (in1Block.getBlockDims()[i] != in2Block.getBlockDims()[i] &&
                in1Block.getBlockDims()[i] != uninitNum && in2Block.getBlockDims()[i] != uninitNum)
            return false;
        if (in1Block.getOffsetPaddingToData()[i] != in2Block.getOffsetPaddingToData()[i] &&
                in1Block.getOffsetPaddingToData()[i] != uninitNum && in2Block.getOffsetPaddingToData()[i] != uninitNum)
            return false;
        if (i >= batch1 && in1Block.getStrides()[i] != in2Block.getStrides()[i] &&
                in1Block.getStrides()[i] != uninitNum && in2Block.getStrides()[i] != uninitNum)
            return false;
        if (in1Block.getOrder()[i] != in2Block.getOrder()[i] &&
                in1Block.getOrder()[i] != uninitNum && in2Block.getOrder()[i] != uninitNum)
            return false;
    }
    return !(in1Block.getOffsetPadding() != in2Block.getOffsetPadding() &&
        in1Block.getOffsetPadding() != uninitNum && in2Block.getOffsetPadding() != uninitNum);
}
