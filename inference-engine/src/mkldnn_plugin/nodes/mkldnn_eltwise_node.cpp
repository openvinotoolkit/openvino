// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include <map>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
    op = EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isSum() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    return eltwiseLayer->_operation == EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isUnitScales() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer->coeff.empty())
        return true;

    for (auto scale : eltwiseLayer->coeff) {
        if (scale != 1.0f)
            return false;
    }

    return true;
}

void MKLDNNEltwiseNode::getSupportedDescriptors() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert eltwise layer.";
    op = eltwiseLayer->_operation;

    if (getParentEdges().size() < 2)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    if (op == EltwiseLayer::Squared_diff)
        if (getParentEdges().size() != 2)
            THROW_IE_EXCEPTION  << "Incorrect number of input edges for layer " << getName() << " for operation squared_diff.\n"
                << "Expected: 2\n" << "Actual: " << getParentEdges().size();

    auto outDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto inDims = getParentEdgeAt(i)->getDims();
        for (size_t j = 1; j <= inDims.ndims(); j++) {
            if (outDims[outDims.ndims() - j] != inDims[inDims.ndims() - j]) {
                if (inDims[inDims.ndims() - j] == 1) {
                    broadcast = true;
                } else {
                    THROW_IE_EXCEPTION << "Incorrect dimentions for broadcasting for " << eltwiseLayer->name;
                }
            }
        }
    }

    if (broadcast) {
        auto outDims = getChildEdgeAt(0)->getDims();
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto inDims = getParentEdgeAt(i)->getDims();
            if (inDims.ndims() > 5 || outDims.ndims() > 5)
                THROW_IE_EXCEPTION << "Eltwise node in broadcasting mode doesn't support more than 5 dims for blobs";
        }
    }

    bool with_coeffs = !eltwiseLayer->coeff.empty();
    if (op != EltwiseLayer::Sum && with_coeffs)
        THROW_IE_EXCEPTION << "Only sum operation supports operands coefficients";

    if (with_coeffs && eltwiseLayer->coeff.size() != getParentEdges().size())
        THROW_IE_EXCEPTION << "Number of provided coefficients is not equal to number of operands";

    if (with_coeffs && eltwiseLayer->precision != Precision::FP32)
        THROW_IE_EXCEPTION << "Sum with coefficients supports only FP32 precision";

    sum_scales.clear();
    for (int i = 0; i < getParentEdges().size(); i++)
        sum_scales.push_back(with_coeffs ? eltwiseLayer->coeff[i] : 1.0f);
}

void MKLDNNEltwiseNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto initDesc = [&] (mkldnn::memory::data_type inputDT, mkldnn::memory::data_type outputDT, memory::format format) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInPlace()) ? 0 : -1;
            dataConfig.constant = false;

            if (getParentEdgeAt(i)->getDims().ndims() == getChildEdgeAt(0)->getDims().ndims()) {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT, format);
                config.inConfs.push_back(dataConfig);
            } else {
                // Broadcasting support
                if (MKLDNNMemory::IsPlainFormat(format)) {
                    dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT, MKLDNNMemory::GetPlainFormat(getParentEdgeAt(i)->getDims()));
                    config.inConfs.push_back(dataConfig);
                }
            }
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDT, format);
            config.outConfs.push_back(dataConfig);
        return {config, impl_desc_type::ref};
    };

    for (const auto& format : getAvailableFormatsForDims(getChildEdgeAt(0)->getDims())) {
        mkldnn::memory::data_type inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->precision);
        mkldnn::memory::data_type outputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->precision);
        supportedPrimitiveDescriptors.push_back(initDesc(inputDT, outputDT, format));
    }
}

void MKLDNNEltwiseNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate.";
        }

        if (op == EltwiseLayer::Sum) {
            srcs_pd.push_back(srcMemPtr->GetPrimitiveDescriptor());
            srcs_p.emplace_back(srcMemPtr->GetPrimitive());
        }
    }
    if (op == EltwiseLayer::Sum && !broadcast) {
        try {
            auto primitive_desc = mkldnn::sum::primitive_desc(dstMemPtr->GetDescriptor(), sum_scales, srcs_pd);
            prim = std::shared_ptr<mkldnn::sum>(new mkldnn::sum(primitive_desc, srcs_p, dstMemPtr->GetPrimitive()));
        } catch (...) {
            std::cerr << "Handle this problem correctly!" << std::endl;
            prim = nullptr;
        }
    }
}

void MKLDNNEltwiseNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    MKLDNNNode::initOptimalPrimitiveDescriptor();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    auto& selectedConfig = getSelectedPrimitiveDescriptor()->getConfig();
    for (size_t i = 1; i < selectedConfig.inConfs.size(); i++) {
        if (selectedConfig.inConfs[0].desc.getPrecision() != selectedConfig.inConfs[i].desc.getPrecision()) {
            selectedConfig.inConfs[i].desc.setPrecision(selectedConfig.inConfs[0].desc.getPrecision());
        }
    }
}

void MKLDNNEltwiseNode::dims_calc(int *dims, const MKLDNNDims &edge_dims) {
    for (int i = 0; i < 5; i++)
        dims[i] = 1;
    int ndims = edge_dims.ndims();
    if (ndims > 5) {
        THROW_IE_EXCEPTION << "ndims should be less then 5";
    }
    for (int i = 0; i < ndims; i++) {
        dims[4 - i] = edge_dims[ndims - 1 - i];
    }
    dims[5 - ndims] = std::min(dims[5 - ndims], batchToProcess());
}

void MKLDNNEltwiseNode::offset_out_calc(int *offset, int *dims) {
    int k = 1;
    for (int i = 4; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

void MKLDNNEltwiseNode::offset_in_calc(int *offset, int *dims_in, int *dims_out) {
    int k = 1;
    for (int i = 4; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

// Intel C++ Compiler 18.0 for Windows contains bug that doesn't allow to use templates to generate eltwise implementations
// and to avoid all copypaste below
template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_add(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] + src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] + src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] + src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] + src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_prod(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] * src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] * src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] * src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] * src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_max(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::max(src0_ptr[i], (T0)src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::max(src0_ptr[i], (T0)src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::max(dst_ptr[i], (T0)src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::max(dst_ptr[i], (T0)src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::max(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = std::max(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::max(dst_ptr[index_out], (T0)src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::max(dst_ptr[index_out], (T0)src_ptr[index_in]);
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_sub(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] - src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] - src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] - src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] - src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] - src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] - src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] - src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] - src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_min(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::min(src0_ptr[i], (T0)src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::min(src0_ptr[i], (T0)src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::min(dst_ptr[i], (T0)src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::min(dst_ptr[i], (T0)src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::min(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = std::min(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::min(dst_ptr[index_out], (T0)src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::min(dst_ptr[index_out], (T0)src_ptr[index_in]);
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_div(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] / src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] / src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] / src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] / src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] / src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] / src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] / src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] / src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_squared_diff(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = (src0_ptr[i] - src1_ptr[i]) * (src0_ptr[i] - src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = (src0_ptr[i] - src1_ptr[i]) * (src0_ptr[i] - src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = (dst_ptr[i] - src_ptr[i]) * (dst_ptr[i] - src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = (dst_ptr[i] - src_ptr[i]) * (dst_ptr[i] - src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = (src0_ptr[index_in0] - src1_ptr[index_in1]) * (src0_ptr[index_in0] - src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = (src0_ptr[index_in0] - src1_ptr[index_in1]) * (src0_ptr[index_in0] - src1_ptr[index_in1]);
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = (dst_ptr[index_out] - src_ptr[index_in]) * (dst_ptr[index_out] - src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = (dst_ptr[index_out] - src_ptr[index_in]) * (dst_ptr[index_out] - src_ptr[index_in]);
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_floor_mod(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] - src0_ptr[i] / src1_ptr[i] * src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] - src0_ptr[i] / src1_ptr[i] * src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] - dst_ptr[i] / src_ptr[i] * src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] - dst_ptr[i] / src_ptr[i] * src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] - src0_ptr[index_in1] / src1_ptr[index_in0] * src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] - src0_ptr[index_in1] / src1_ptr[index_in0] * src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] - dst_ptr[index_in] / src_ptr[index_out] * src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] - dst_ptr[index_in] / src_ptr[index_out] * src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_pow(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::pow(src0_ptr[i], src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::pow(src0_ptr[i], src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::pow(dst_ptr[i], src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::pow(dst_ptr[i], src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::pow(src0_ptr[index_in0], src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = std::pow(src0_ptr[index_in0], src1_ptr[index_in1]);
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::pow(dst_ptr[index_out], src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::pow(dst_ptr[index_out], src_ptr[index_in]);
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] == src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] == src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] == src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] == src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] == src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] == src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] == src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] == src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_not_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] != src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] != src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] != src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] != src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] != src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] != src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] != src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] != src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_less(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] < src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] < src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] < src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] < src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] < src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] < src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] < src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] < src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_less_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] <= src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] <= src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] <= src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] <= src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] <= src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] <= src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] <= src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] <= src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_greater(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] > src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] > src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] > src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] > src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] > src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] > src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] > src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] > src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_greater_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] >= src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] >= src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] >= src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] >= src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] >= src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] >= src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] >= src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] >= src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_and(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] && src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] && src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] && src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] && src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] && src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] && src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] && src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] && src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_or(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] || src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] || src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] || src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] || src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] || src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = src0_ptr[index_in0] || src1_ptr[index_in1];
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] || src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] || src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_xor(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = (src0_ptr[i] || src1_ptr[i]) - (src0_ptr[i] && src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = (src0_ptr[i] || src1_ptr[i]) - (src0_ptr[i] && src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = (dst_ptr[i] || src_ptr[i]) - (dst_ptr[i] && src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = (dst_ptr[i] || src_ptr[i]) - (dst_ptr[i] && src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = (src0_ptr[index_in0] || src1_ptr[index_in1]) - (src0_ptr[index_in0] && src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
            dst_ptr[index_out] = (src0_ptr[index_in0] || src1_ptr[index_in1]) - (src0_ptr[index_in0] && src1_ptr[index_in1]);
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = (dst_ptr[index_out] || src_ptr[index_in]) - (dst_ptr[index_out] && src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = (dst_ptr[index_out] || src_ptr[index_in]) - (dst_ptr[index_out] && src_ptr[index_in]);
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::ref_eltwise(int in0, int in1) {
    IE_ASSERT(getParentEdges().size() > 1);

    auto& srcMemory0 = getParentEdgeAt(in0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(in1)->getMemory();
    const T0 *src0_ptr = reinterpret_cast<const T0*>(srcMemory0.GetData()) +
            srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const T1 *src1_ptr = reinterpret_cast<const T1*>(srcMemory1.GetData()) +
            srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    T0 *dst_ptr = reinterpret_cast<T0*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    const size_t dst_data_size = srcMemory0.GetSize() / sizeof(T0) / srcMemory0.GetDims()[0] * batchToProcess();

    switch (op) {
        case EltwiseLayer::eOperation::Sum: eltwise_add(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Prod: eltwise_prod(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Max: eltwise_max(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Sub: eltwise_sub(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Min: eltwise_min(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Div: eltwise_div(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Squared_diff: eltwise_squared_diff(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Floor_mod: eltwise_floor_mod(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Pow: eltwise_pow(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Equal: eltwise_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Not_equal: eltwise_not_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less: eltwise_less(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less_equal: eltwise_less_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater: eltwise_greater(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater_equal: eltwise_greater_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_AND: eltwise_logical_and(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_OR: eltwise_logical_or(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_XOR: eltwise_logical_xor(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node";
    }
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (prim) {
        MKLDNNNode::execute(strm);
    } else {
        if (op == EltwiseLayer::Floor_mod) {
            for (size_t i = 0; i < getParentEdges().size(); i++)
                if (getParentEdgeAt(i)->getDesc().getPrecision() != Precision::I32)
                    THROW_IE_EXCEPTION << "Floor_mod supports only I32 precision of inputs";
            if (getChildEdgeAt(0)->getDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << "Floor_mod supports only I32 precision of output";
        }
        if (getParentEdges().size() > 2) {
            Precision pi = getParentEdgeAt(0)->getDesc().getPrecision();
            Precision po = getChildEdgeAt(0)->getDesc().getPrecision();
            for (int i = 1; i < getParentEdges().size(); i++) {
                if (getParentEdgeAt(i)->getDesc().getPrecision() != pi)
                    THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, all inputs must have same precision";
            }
            if (pi != po) {
                THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, all inputs and output must have same precision";
            }
            if (pi == Precision::FP32)
                ref_eltwise<float, float>(0, 1);
            else if (pi == Precision::I32)
                ref_eltwise<int32_t, int32_t>(0, 1);
            else if (pi == Precision::I8)
                ref_eltwise<int8_t, int8_t>(0, 1);
            else if (pi == Precision::U8)
                ref_eltwise<uint8_t, uint8_t>(0, 1);
            else
                THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, only FP32, I32, I8, U8 are supported";
            return;
        }

        Precision pi0 = getParentEdgeAt(0)->getDesc().getPrecision();
        Precision pi1 = getParentEdgeAt(1)->getDesc().getPrecision();
        Precision po = getChildEdgeAt(0)->getDesc().getPrecision();

        IE_ASSERT(getParentEdges().size() > 1);

        if (po == Precision::FP32 && pi0 == po && pi1 == po) {
            ref_eltwise<float, float>(0, 1);
        } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::I8) {
            ref_eltwise<float, int8_t>(0, 1);
        } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::I8) {
            ref_eltwise<float, int8_t>(1, 0);
        } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::U8) {
            ref_eltwise<float, uint8_t>(0, 1);
        } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::U8) {
            ref_eltwise<float, uint8_t>(1, 0);
        } else if (po == Precision::I8 && pi0 == po && pi1 == po) {
            ref_eltwise<int8_t, int8_t>(0, 1);
        } else if (po == Precision::I8 && pi0 == po && pi1 == Precision::U8) {
            ref_eltwise<int8_t, uint8_t>(0, 1);
        } else if (po == Precision::I8 && pi1 == po && pi0 == Precision::U8) {
            ref_eltwise<int8_t, uint8_t>(1, 0);
        } else if (po == Precision::I32 && pi0 == po && pi1 == po) {
            ref_eltwise<int32_t, int32_t>(0, 1);
        }
    }
}

bool MKLDNNEltwiseNode::created() const {
    return getType() == Eltwise;
}

bool MKLDNNEltwiseNode::canBeInPlace() const {
    size_t inPlaceWithParent = getParentEdges().size();
    for (size_t i = 0; i < inPlaceWithParent; i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (!parentEdge->getParent()->isConstant() &&
                parentEdge->getParent()->getChildEdges().size() == 1) {
            inPlaceWithParent = i;
            break;
        }
    }
    // This is WA for MKLDNN implementation
    if (inPlaceWithParent != 0)
        return false;
    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }

    return true;
}
