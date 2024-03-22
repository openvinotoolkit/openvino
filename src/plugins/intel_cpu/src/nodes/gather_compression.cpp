// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_compression.h"

#include "common/cpu_memcpy.h"
#include "ov_ops/gather_compressed.hpp"
#include "utils/ngraph_utils.hpp"

using namespace dnnl::impl::cpu;

#define THROW_ERROR(...) OPENVINO_THROW(getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherCompression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gather_compression = std::dynamic_pointer_cast<const ov::op::internal::GatherCompressed>(op);
        if (!gather_compression) {
            errorMessage = "Only GatherCompression operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GatherCompression::GatherCompression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if ((op->get_input_size() != 4u && op->get_input_size() != 5u) || op->get_output_size() != 1u)
        THROW_ERROR("has incorrect number of input/output[",
                    op->get_input_size(),
                    ",",
                    op->get_output_size(),
                    "] edges!");
}

void GatherCompression::initSupportedPrimitiveDescriptors() {
    std::cout << "--->2:GatherCompression::initSupportedPrimitiveDescriptors() --->\n";
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    if (!one_of(dataPrecision, ov::element::u8, ov::element::u4, ov::element::i8, ov::element::i4)) {
        THROW_ERROR("has unsupported 'data' input precision: ", dataPrecision);
    }

    ov::element::Type scalePrecision = getOriginalInputPrecisionAtPort(GATHER_SCALE);
    if (scalePrecision != ov::element::f32) {
        THROW_ERROR("has unsupported 'scale' input precision: ", scalePrecision);
    }

    ov::element::Type outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!one_of(outPrecision, ov::element::f32, ov::element::f16)) {
        THROW_ERROR("has unsupported out precision: ", outPrecision);
    }

    if (getOriginalInputsNumber() == 5u) {
        ov::element::Type zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
        if (zpPrecision != ov::element::f32) {
            THROW_ERROR("has unsupported 'zp' input precision: ", zpPrecision);
        }
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, scalePrecision},
                              {LayoutType::ncsp, zpPrecision}},
                             {{LayoutType::ncsp, outPrecision}},
                             ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, scalePrecision}},
                             {{LayoutType::ncsp, outPrecision}},
                             ref_any);
    }
}

bool GatherCompression::needPrepareParams() const {
    return false;
}

void GatherCompression::execute(dnnl::stream strm) {
    execReference();
}

void GatherCompression::executeDynamicImpl(dnnl::stream strm) {
    execReference();
}

template <typename OUT_PRECISION>
void GatherCompression::execReferenceU4() {
    std::cout << "--->5:GatherCompression::execReferenceU4()" << std::endl;
}

template <typename OUT_PRECISION>
void GatherCompression::execReferenceI4() {
    std::cout << "--->5:GatherCompression::execReferenceI4()" << std::endl;
    DEBUG_LOG(getName(), "execReference4bit");
    auto data_mem_ptr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    auto ind_mem_ptr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    const auto* psrc = data_mem_ptr->getDataAs<uint8_t>();
    const auto* pidx = ind_mem_ptr->getDataAs<int32_t>();

    bool one_dim_zp = getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape().getRank() == 1;
    const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<float>(0);

    const auto& idxDims = ind_mem_ptr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    auto axisDim = data_mem_ptr->getStaticDims()[0];
    auto groupDim = data_mem_ptr->getStaticDims().size() == 2 ? 1 : data_mem_ptr->getStaticDims()[1];
    auto feaDim = data_mem_ptr->getStaticDims().size() == 2 ? data_mem_ptr->getStaticDims()[1] : data_mem_ptr->getStaticDims()[2];

    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto* dst = pdst + dstIdx * feaDim * groupDim;
        auto* src = psrc + ii * feaDim * groupDim / 2;

        for (size_t g = 0; g < groupDim; g++) {
            // auto& deq_zp = zp[ii];
            // auto& deq_scale = scale[ii];
            auto& deq_zp = one_dim_zp ? zp[0] : zp[ii * groupDim + g];
            auto& deq_scale = scale[ii * groupDim + g];

            size_t k = 0;
            for (; k < feaDim; k += 2) {
                auto x = src[0];
                dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
                dst[1] = ((x >> 4) - deq_zp) * deq_scale;
                dst += 2;
                src++;
            }
            // Process last one if feaDim is odd
            for (; k < feaDim; k++) {
                auto x = src[0];
                dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
                dst++;
                src++;
            }
        }
    });
}

std::string shape2str(ov::intel_cpu::Shape shape) {
    std::string str = "[";
    for (auto s : shape.getStaticDims()) {
        str += std::to_string(s) + ",";
    }
    return str + "]";
}

template <typename IN_PRECISION, typename OUT_PRECISION>
void GatherCompression::execReference8bit() {
    DEBUG_LOG(getName(), "execReference8bit");
    std::cout << "--->4:GatherCompression::execReference8bit()\n";
#define PRINT(X) std::cout << #X << " = " << X << std::endl

    auto data_mem_ptr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    auto ind_mem_ptr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    auto scale_mem_ptr = getParentEdgeAt(GATHER_SCALE)->getMemoryPtr();
    const auto* psrc = data_mem_ptr->getDataAs<IN_PRECISION>();
    PRINT(ind_mem_ptr->getPrecision());
    PRINT(getChildEdgeAt(0)->getMemoryPtr()->getPrecision());
    PRINT(shape2str(getChildEdgeAt(0)->getMemoryPtr()->getShape()));
    const auto* pidx = ind_mem_ptr->getDataAs<int32_t>();

    bool have_zp = getOriginalInputsNumber() > 4u;
    auto check_one_dim = [](MemoryPtr mem_ptr, bool& is_const_scale) {
        const auto& shape = mem_ptr->getStaticDims();
        if (shape.size() == 1 && shape[0] == 1) {
            is_const_scale = true;
            return true;
        } else if (shape.size() == 3 && shape[1] != 1) {
            return false;
        }
        return true;
    };
    bool is_const_zp = false;
    bool one_dim_zp = have_zp ? check_one_dim(getParentEdgeAt(GATHER_ZP)->getMemoryPtr(), is_const_zp) : true;
    bool is_const_scale = false;
    bool one_dim_scale = check_one_dim(scale_mem_ptr, is_const_scale);

    PRINT(have_zp);
    PRINT(is_const_zp);
    PRINT(one_dim_zp);

    PRINT(is_const_scale);
    PRINT(one_dim_scale);

    float_t* zp = nullptr;
    float_t const_zp = 0.f;
    zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<OUT_PRECISION>(0);

    const auto& idxDims = ind_mem_ptr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    PRINT(batch);
    PRINT(seqLen);
    if (have_zp)
        PRINT(shape2str(getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape()));
    PRINT(shape2str(scale_mem_ptr->getShape()));

    auto axisDim = data_mem_ptr->getStaticDims()[0];
    auto groupDim = data_mem_ptr->getStaticDims().size() == 2 ? 1 : data_mem_ptr->getStaticDims()[1];
    auto feaDim =
        data_mem_ptr->getStaticDims().size() == 2 ? data_mem_ptr->getStaticDims()[1] : data_mem_ptr->getStaticDims()[2];

    PRINT(shape2str(data_mem_ptr->getShape()));
    PRINT(axisDim);
    PRINT(groupDim);
    PRINT(feaDim);

#if 0
    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto* src = psrc + ii * feaDim * groupDim;
        auto* dst = pdst + dstIdx * feaDim * groupDim;

        for (size_t g = 0; g < groupDim; g++) {
            auto& deq_zp = one_dim_zp ? zp[0] : zp[ii * groupDim + g];
            auto& deq_scale = scale[ii * groupDim + g];
            for (size_t k = 0; k < feaDim; k++) {
                dst[0] = static_cast<OUT_PRECISION>((static_cast<float>(src[0]) - deq_zp) * deq_scale);
                dst++;
                src++;
            }
        }
    });
#else
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seqLen; s++) {
            auto dstIdx = b * seqLen + s;
            auto ii = pidx[dstIdx];
            if (ii < 0) {
                if (reverseIndexing)
                    ii += axisDim;
                else
                    ii = axisDim;
            }

            auto* src = psrc + ii * feaDim * groupDim;
            auto* dst = pdst + dstIdx * feaDim * groupDim;

            for (size_t g = 0; g < groupDim; g++) {
                auto& deq_zp = one_dim_zp ? (have_zp ? zp[ii] : zp[0]) : zp[ii * groupDim + g];
                auto& deq_scale = one_dim_scale ? scale[ii] : scale[ii * groupDim + g];
                for (size_t k = 0; k < feaDim; k++) {
                    // dst[0] = static_cast<OUT_PRECISION>((static_cast<float>(src[0]) - deq_zp) * deq_scale);
                    dst++;
                    src++;
                }
            }
        }
    }
#endif
}

void GatherCompression::execReference() {
    auto in_precison = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getPrecision();
    auto out_precision = getChildEdgeAt(0)->getMemoryPtr()->getPrecision();

    if (out_precision == ov::element::f16) {
        switch (in_precison) {
        case ov::element::u8:
            return execReference8bit<uint8_t, float16>();
        case ov::element::i8:
            return execReference8bit<int8_t, float16>();
        case ov::element::u4:
            return execReferenceU4<float16>();
        case ov::element::i4:
            return execReferenceI4<float16>();
        default:
            break;
        }
    } else if (out_precision == ov::element::f32) {
        switch (in_precison) {
        case ov::element::u8:
            return execReference8bit<uint8_t, float>();
        case ov::element::i8:
            return execReference8bit<int8_t, float>();
        case ov::element::u4:
            return execReferenceU4<float>();
        case ov::element::i4:
            return execReferenceI4<float>();
        default:
            break;
        }
    }

    THROW_ERROR("only support in precision(u4/i4/u8/i8), out precision(f32/f16), in_precison=",
                in_precison,
                ", out_precision=",
                out_precision);
}

bool GatherCompression::created() const {
    return getType() == Type::GatherCompression;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov