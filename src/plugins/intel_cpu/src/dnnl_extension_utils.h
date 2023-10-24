// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN memory formats.
 * @file dnnl_extension_utils.h
 */
#pragma once

#include <string>

#include "onednn/dnnl.h"
#include "memory_desc/cpu_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include <common/c_types_map.hpp>

namespace ov {
namespace intel_cpu {

class DnnlMemoryDesc;

class DnnlExtensionUtils {
public:
    static uint8_t sizeOfDataType(dnnl::memory::data_type dataType);
    static dnnl::memory::data_type IEPrecisionToDataType(const InferenceEngine::Precision& prec);
    static InferenceEngine::Precision DataTypeToIEPrecision(dnnl::memory::data_type dataType);
    static dnnl::memory::data_type ElementTypeToDataType(const ov::element::Type& elementType);
    static ov::element::Type DataTypeToElementType(const dnnl::memory::data_type& dataType);
    static Dim convertToDim(const dnnl::memory::dim &dim);
    static dnnl::memory::dim convertToDnnlDim(const Dim &dim);
    static VectorDims convertToVectorDims(const dnnl::memory::dims& dims);
    static VectorDims convertToVectorDims(const dnnl::impl::dims_t dims, const int ndims);
    static std::vector<dnnl::memory::dim> convertToDnnlDims(const VectorDims& dims);
    static dnnl::memory::format_tag GetPlainFormatByRank(size_t rank);

    /**
     * @brief Creates DnnlBlockedMemoryDesc if desc is blocked, otherwise DnnlMemoryDesc
     * @param desc dnnl::memory::desc from which one of the descriptors will be created
     * @return pointer to DnnlBlockedMemoryDesc or DnnlMemoryDesc
     */
    static std::shared_ptr<DnnlMemoryDesc> makeDescriptor(const dnnl::memory::desc &desc);
    static std::shared_ptr<DnnlMemoryDesc> makeDescriptor(const_dnnl_memory_desc_t desc);

    /**
     * @brief Helper function that creates DnnlBlockedMemoryDesc from defined dnnl::memory::desc and undefined shape.
     * It uses desc as an basis for the new undefined one. Specifically, type, layout, precision, blocks, extra data will be preserved.
     * @param desc dnnl::memory::desc dnnl desc which will be used as a basis of the new descriptor
     * @param shape a new undefined shape
     * @return pointer to the created DnnlBlockedMemoryDesc
     * @note Obly blocked descriptors are allowed at the moment
     */

    static std::shared_ptr<DnnlBlockedMemoryDesc> makeUndefinedDesc(const dnnl::memory::desc &desc, const Shape& shape);
    static size_t getMemSizeForDnnlDesc(const dnnl::memory::desc& desc);

    static std::shared_ptr<DnnlMemoryDesc> query_md(const const_dnnl_primitive_desc_t& pd, const dnnl::query& what, int idx = 0);
    static std::string query_impl_info_str(const const_dnnl_primitive_desc_t& pd);

    template<typename T>
    static bool find_implementation(dnnl::primitive_desc& desc, T&& comparator) {
        dnnl::primitive_desc_iterator& itpd = desc;

        while (itpd) {
            const impl_desc_type descImplType = parse_impl_name(itpd.impl_info_str());

            if (comparator(descImplType)) {
                return true;
            }

            if (!itpd.next_impl())
                break;
        }

        return false;
    }

    template<typename T, typename L>
    static void for_each_implementation(dnnl::primitive_desc& desc, bool first_match, T&& comparator, L&& func) {
        dnnl::primitive_desc_iterator& itpd = desc;

        while (itpd) {
            const impl_desc_type descImplType = parse_impl_name(itpd.impl_info_str());

            if (comparator(descImplType)) {
                func(itpd);
                if (first_match)
                    break;
            }

            if (!itpd.next_impl())
                break;
        }

        return;
    }

    static bool find_implementation(dnnl::primitive_desc& desc, impl_desc_type implType);
    static dnnl_primitive_desc_t clone_primitive_desc(const_dnnl_primitive_desc_t cprim_desc);
    static dnnl_memory_desc_t clone_desc(const_dnnl_memory_desc_t cdesc);
    static const char* query_pd_info(const_dnnl_primitive_desc_t pd);
    static dnnl::algorithm convertToDnnlAlgorithm(Algorithm alg);
    static bool isUnarySupportedAsPostOp(Algorithm alg);
};

}   // namespace intel_cpu
}   // namespace ov
