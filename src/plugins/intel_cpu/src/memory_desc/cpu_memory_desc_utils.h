// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "cpu_shape.h"
#include "cpu_types.h"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "graph_context.h"

namespace ov {
namespace intel_cpu {

class MemoryDesc;
class DnnlMemoryDesc;
class BlockedMemoryDesc;
class DnnlBlockedMemoryDesc;
class CpuBlockedMemoryDesc;
class EmptyMemoryDesc;
class IMemory;
class Memory;

class MemoryDescUtils {
public:
    MemoryDescUtils() = delete;

    /**
     * @brief Converts MemoryDesc to DnnlMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted DnnlMemoryDesc
     */
    static std::shared_ptr<DnnlMemoryDesc> convertToDnnlMemoryDesc(const std::shared_ptr<MemoryDesc> &desc);

    /**
     * @brief Converts MemoryDesc to DnnlBlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted DnnlBockedMemoryDesc
     */
    static DnnlBlockedMemoryDesc convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts MemoryDesc to BlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted BlockedMemoryDesc
     */
    static std::shared_ptr<BlockedMemoryDesc> convertToBlockedMemoryDesc(const std::shared_ptr<MemoryDesc> &desc);

    /**
     * @brief Builds CpuBlockedMemoryDesc for given ov::ITensor
     * @param tensor is a pointer to ov::ITensor object
     * @return created CpuBlockedMemoryDesc
     */
    static std::shared_ptr<CpuBlockedMemoryDesc> generateCpuBlockedMemoryDesc(const ov::SoPtr<ov::ITensor>& tensor);

    static constexpr Dim DEFAULT_DUMMY_VAL = 64;

    /**
     * @brief Makes a dummy descriptor where all undefined values are replaced with the smallest value between the parameter and the upper bound dim
     * @param desc MemoryDesc from which the new descriptor is generated
     * @param dummyVal Dim value to replace undefined dimensions
     * @return a new MemoryDesc with dummy values instead of undefined dims
     */
    static std::shared_ptr<MemoryDesc> makeDummyDesc(const MemoryDesc& desc, Dim dummyVal = DEFAULT_DUMMY_VAL);

    /**
    * @brief Make an empty memory descriptor
    * @note Shape{0}, undefined
    * @return empty memory descriptor
    */
    static std::shared_ptr<MemoryDesc> makeEmptyDesc();
    static std::shared_ptr<IMemory> makeEmptyMemory(const GraphContext::CPtr context);

    /**
    * @brief Makes a static dummy shape where all undefined values are replaced with the smallest value between the parameter and the upper bound dim
    * @param shape a Shape object from which the new static shape is generated
    * @param dummyVal Dim value to replace undefined dimensions
    * @return a new Shape with dummy values instead of undefined dims
    */
    static Shape makeDummyShape(const Shape& shape, Dim dummyVal = DEFAULT_DUMMY_VAL);

    /**
    * @brief Makes a static dummy shape where all undefined values are replaced with the smallest value between the parameter and the upper bound dim
    * @param shape a Shape object from which the new static shape is generated
    * @param dummyVals vector of values to replace undefined dimensions
    * @return a new Shape with dummy values instead of undefined dims
    */
    static Shape makeDummyShape(const Shape& shape, const VectorDims& dummyVals);

    /**
     * @brief Converts dim to string, undefined dim represented as ?
     * @param dim Dim to be converted
     * @return dim as string
     */
    static std::string dim2str(Dim dim);

    /**
     * @brief Converts dims to string, undefined dim represented as ?
     * @param dim Dims to be converted
     * @return dims as string
     */
    static std::string dims2str(const VectorDims& dims);
};

}   // namespace intel_cpu
}   // namespace ov
