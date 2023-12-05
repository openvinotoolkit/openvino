// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "dump.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "backend/gna_limitations.hpp"
#include "gna2-model-api.h"
#include "gna2_model_helper.hpp"
#include "gna_device.hpp"
#include "log.hpp"

using namespace ov::intel_gna::limitations;

namespace ov {
namespace intel_gna {
namespace dump {

inline std::string GetLayerType(Gna2OperationType type) {
    switch (type) {
    case Gna2OperationTypeFullyConnectedAffine:
        return "Gna2OperationTypeFullyConnectedAffine";
    case Gna2OperationTypeElementWiseAffine:
        return "Gna2OperationTypeElementWiseAffine";
    case Gna2OperationTypeRecurrent:
        return "Gna2OperationTypeRecurrent";
    case Gna2OperationTypeConvolution:
        return "Gna2OperationTypeConvolution";
    case Gna2OperationTypeTransposition:
        return "Gna2OperationTypeTransposition";
    case Gna2OperationTypeCopy:
        return "Gna2OperationTypeCopy";
    default:
        return "Gna2OperationUNKNOWN";
    }
}

inline void write_pwl(std::ostream& pwl_file, const Gna2PwlSegment* const segments, const uint32_t numberOfSegments) {
    for (uint32_t k = 0; k < numberOfSegments; k++) {
        pwl_file << segments[k].Slope << ", " << segments[k].xBase << ", " << segments[k].yBase << "\n";
    }
}

inline void write_pwl(std::ostream& pwl_file, const Gna2Tensor& activation) {
    write_pwl(pwl_file, static_cast<Gna2PwlSegment*>(activation.Data), activation.Shape.Dimensions[0]);
}

inline std::string GetSimpleString(Gna2Shape shape) {
    std::stringstream out;
    for (uint32_t i = 0; i < shape.NumberOfDimensions; i++) {
        out << shape.Dimensions[i];
        if (i + 1 < shape.NumberOfDimensions)
            out << 'x';
    }
    return out.str();
}

template <class MapType>
uint32_t FindInMapOrReturnOne(MapType map, typename MapType::key_type key) {
    auto value = map.find(key);
    if (value != map.end()) {
        return value->second;
    }
    return 1;
}

inline uint32_t GetTypeByteSize(Gna2DataType type) {
    static const std::map<Gna2DataType, uint32_t> operandTypeMap = {{Gna2DataTypeNone, 1},
                                                                    {Gna2DataTypeBoolean, 1},
                                                                    {Gna2DataTypeInt4, 1},
                                                                    {Gna2DataTypeInt8, 1},
                                                                    {Gna2DataTypeInt16, 2},
                                                                    {Gna2DataTypeInt32, 4},
                                                                    {Gna2DataTypeUint4, 1},
                                                                    {Gna2DataTypeUint8, 1},
                                                                    {Gna2DataTypeUint16, 2},
                                                                    {Gna2DataTypeUint32, 4},
                                                                    {Gna2DataTypeUint64, 8},
                                                                    {Gna2DataTypeCompoundBias, 8},
                                                                    {Gna2DataTypePwlSegment, 8},
                                                                    {Gna2DataTypeWeightScaleFactor, 8}};
    return FindInMapOrReturnOne(operandTypeMap, type);
}

inline uint32_t GetGnaShapeSize(const Gna2Shape& shape, const uint32_t bytesPerElement) {
    if (shape.NumberOfDimensions == 0) {
        return 0;
    }
    // to compute aligned filters (each filter begin is aligned to 16B)
    // e.g., for 3x3 2B filter, its size is 18B, but the next filter will start at 32B offset
    // filters are NHWC
    uint32_t nAlignement = 1;
    if (shape.NumberOfDimensions == 4 && shape.Dimensions[0] != 1) {
        nAlignement = 16;
    }
    uint32_t total = 1;
    for (uint32_t i = 1; i < shape.NumberOfDimensions; i++) {
        total *= shape.Dimensions[i];
    }
    total *= bytesPerElement;
    auto totalAligned = Gna2RoundUp(total, nAlignement);
    totalAligned *= shape.Dimensions[0];
    return totalAligned;
}

template <class T>
bool NextElement(T& elementIndex, const Gna2Shape& total) {
    if (total.NumberOfDimensions == 0)
        return false;
    auto idx = total.NumberOfDimensions - 1;
    IE_ASSERT(idx < GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS);
    while (elementIndex[idx] + 1 >= total.Dimensions[idx] && idx > 0) {
        idx--;
    }
    if (elementIndex[idx] + 1 >= total.Dimensions[idx] && idx == 0) {
        return false;
    }
    ++(elementIndex[idx]);
    while (++idx < total.NumberOfDimensions) {
        elementIndex[idx] = 0;
    }
    return true;
}

template <class T>
uint32_t GetLinearIndex(const T& elementIndex, const Gna2Shape& total) {
    uint32_t out = 0;
    for (uint32_t idx = 0; idx < total.NumberOfDimensions; idx++) {
        out += elementIndex[idx];
        if (idx + 1 < total.NumberOfDimensions)
            out *= total.Dimensions[idx + 1];
    }
    return out;
}

template <class T>
int32_t GetValue(const Gna2Tensor& tensor, const T& elementIndex) {
    int32_t intValue = -123;  // fake value indicating problem
    const auto linearIndex = GetLinearIndex(elementIndex, tensor.Shape);
    if (tensor.Type == Gna2DataTypeInt32) {
        intValue = (reinterpret_cast<int32_t*>(tensor.Data)[linearIndex]);
    } else if (tensor.Type == Gna2DataTypeInt16) {
        intValue = reinterpret_cast<int16_t*>(tensor.Data)[linearIndex];
    } else {
        intValue = reinterpret_cast<int8_t*>(tensor.Data)[linearIndex];
    }
    return intValue;
}

void WriteInputAndOutputTextGNAImpl(const Gna2Model& gnaModel,
                                    const std::string dumpFolderNameGNA,
                                    const std::string refFolderName) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        const auto& operation = gnaModel.Operations[i];
        if (operation.Type == Gna2OperationTypeNone) {
            continue;
        }
        std::stringstream out_file_name;
        const auto& outputTensor = *operation.Operands[OutOpIdx];
        const auto& intputTensor = *operation.Operands[InOpIdx];

        out_file_name << std::setfill('0') << std::setw(2) << i << "_" << GetLayerType(operation.Type) << "-"
                      << GetSimpleString(intputTensor.Shape) << "-" << GetSimpleString(outputTensor.Shape);

        auto inputfileName = dumpFolderNameGNA + out_file_name.str() + "_input.txt";
        auto outFileName = dumpFolderNameGNA + out_file_name.str() + "_output.txt";
        auto pwlFileName = dumpFolderNameGNA + out_file_name.str() + "_pwl.txt";
        auto refOutputFileName = refFolderName + out_file_name.str() + "_output.txt";

        std::ofstream out_file(outFileName.c_str(), std::ios::out);
        std::ofstream pwl_file(pwlFileName.c_str(), std::ios::out);
        std::ifstream ref_out_file(refOutputFileName.c_str(), std::ios::in);
        std::ofstream in_file(inputfileName.c_str(), std::ios::out);

        float summOfDiff = 0.f;
        float summOfSqDiff = 0.f;
        float maxD = 0.0f;
        int numItems = 0;

        if (operation.NumberOfOperands > PwlOpIdx && operation.Operands[PwlOpIdx] != nullptr) {
            write_pwl(pwl_file, *operation.Operands[PwlOpIdx]);
        }

        std::vector<uint32_t> elementIndex(outputTensor.Shape.NumberOfDimensions);

        do {
            float floatValue = static_cast<float>(GetValue(outputTensor, elementIndex));

            out_file << std::setw(8) << floatValue << "\n";
            if (ref_out_file) {
                float ref_value = 0.f;
                ref_out_file >> ref_value;
                float diff = (ref_value - floatValue);
                diff = diff < 0 ? -diff : diff;
                summOfDiff += diff;
                summOfSqDiff += diff * diff;
                maxD = std::max(maxD, diff);
                numItems++;
            }
        } while (NextElement(elementIndex, outputTensor.Shape));

        if (numItems) {
            auto rmse = std::sqrt(summOfSqDiff / numItems);
            auto avg = summOfDiff / numItems;
            log::debug() << std::left << std::setw(55) << out_file_name.str() << " RMSE=" << std::fixed
                         << std::setprecision(5) << std::right << std::setw(8) << rmse << " avg=" << std::fixed
                         << std::setprecision(5) << std::right << std::setw(8) << avg << " maxD=" << std::fixed
                         << std::setprecision(5) << std::right << std::setw(8) << maxD << std::endl;
        }

        std::vector<uint32_t> inputElementIndex(intputTensor.Shape.NumberOfDimensions);

        do {
            int32_t intValue = GetValue(intputTensor, inputElementIndex);
            in_file << std::setw(8) << intValue;
            in_file << "\n";
        } while (NextElement(inputElementIndex, intputTensor.Shape));
    }
}

template <typename T>
static std::string GetName(T name, size_t index) {
    return name;
}

template <>
std::string GetName<>(std::vector<std::string> names, size_t index) {
    return names.at(index);
}

template <class MapType>
std::string FindInMapOrReturnUnknown(MapType map, typename MapType::key_type key, size_t index = 0) {
    auto value = map.find(key);
    if (value != map.end()) {
        return GetName(value->second, index);
    }
    return std::string{"unknown"};
}

inline std::string GetOperandType(Gna2DataType type) {
    static const std::map<Gna2DataType, std::string> operandTypeMap = {
        {Gna2DataTypeNone, "Gna2DataTypeNone"},
        {Gna2DataTypeBoolean, "Gna2DataTypeBoolean"},
        {Gna2DataTypeInt4, "Gna2DataTypeInt4"},
        {Gna2DataTypeInt8, "Gna2DataTypeInt8"},
        {Gna2DataTypeInt16, "Gna2DataTypeInt16"},
        {Gna2DataTypeInt32, "Gna2DataTypeInt32"},
        {Gna2DataTypeUint4, "Gna2DataTypeUint4"},
        {Gna2DataTypeUint8, "Gna2DataTypeUint8"},
        {Gna2DataTypeUint16, "Gna2DataTypeUint16"},
        {Gna2DataTypeUint32, "Gna2DataTypeUint32"},
        {Gna2DataTypeUint64, "Gna2DataTypeUint64"},
        {Gna2DataTypeCompoundBias, "Gna2DataTypeCompoundBias"},
        {Gna2DataTypePwlSegment, "Gna2DataTypePwlSegment"},
        {Gna2DataTypeWeightScaleFactor, "Gna2DataTypeWeightScaleFactor"}};
    return FindInMapOrReturnUnknown(operandTypeMap, type);
}

inline std::string GetOperandName(Gna2OperationType type, size_t index) {
    static const std::map<Gna2OperationType, std::vector<std::string>> operationOperandNamesMap = {
        {Gna2OperationTypeConvolution, {"inputs", "outputs", "filters", "biases", "activationFunction"}},
        {Gna2OperationTypeCopy, {"inputs", "outputs"}},
        {Gna2OperationTypeFullyConnectedAffine,
         {"inputs", "outputs", "weights", "biases", "activationFunction", "weightScaleFactors"}},
        {Gna2OperationTypeElementWiseAffine, {"inputs", "outputs", "weights", "biases", "activationFunction"}},
        {Gna2OperationTypeGmm, {"inputs", "outputs", "means/interleaved", "inverseCovariances", "constants"}},
        {Gna2OperationTypeRecurrent, {"inputs", "outputs", "weights", "biases", "activationFunction"}},
        {Gna2OperationTypeTransposition, {"inputs", "outputs"}}};
    return FindInMapOrReturnUnknown(operationOperandNamesMap, type, index);
}

inline std::string GetBiasMode(Gna2BiasMode mode) {
    static const std::map<Gna2BiasMode, std::string> biasModeMap = {{Gna2BiasModeDefault, "Gna2BiasModeDefault"},
                                                                    {Gna2BiasModePerStride, "Gna2BiasModePerStride"},
                                                                    {Gna2BiasModeGrouping, "Gna2BiasModeGrouping"}};
    return FindInMapOrReturnUnknown(biasModeMap, mode);
}

inline std::string GetPoolingMode(Gna2PoolingMode mode) {
    static const std::map<Gna2PoolingMode, std::string> poolingModeMap = {
        {Gna2PoolingModeDisabled, "Gna2PoolingModeDisabled"},
        {Gna2PoolingModeMax, "Gna2PoolingModeMax"},
        {Gna2PoolingModeSum, "Gna2PoolingModeSum"}};
    return FindInMapOrReturnUnknown(poolingModeMap, mode);
}

inline void DumpShape(std::ostream& dumpFile, Gna2Shape* shape, const std::string paramName) {
    dumpFile << "\tParameter name: " << paramName << ", ";
    dumpFile << "parameter type: Gna2Shape\n";
    dumpFile << "\t\tNumber of dimensions: " << shape->NumberOfDimensions;
    dumpFile << "\n\t\tDimensions: [";
    for (uint32_t i = 0; i < shape->NumberOfDimensions; i++) {
        dumpFile << std::setw(8) << shape->Dimensions[i];
    }
    dumpFile << "]\n";
}

inline void DumpConvolutionParameters(std::ostream& dumpFile,
                                      void** parameters,
                                      size_t knownParamCount,
                                      const std::vector<std::string> paramNames) {
    size_t i = 0;

    while (i < knownParamCount) {
        if (i == ConvStrideParamIdx || i == PoolWinParamIdx || i == PoolStrideParamIdx || i == ZeroPaddingParamIdx) {
            Gna2Shape* shape = reinterpret_cast<Gna2Shape*>(parameters[i]);
            if (shape != nullptr)
                DumpShape(dumpFile, shape, paramNames[i]);
        } else if (i == BiasModeCnnParamIdx) {
            Gna2BiasMode* biasMode = reinterpret_cast<Gna2BiasMode*>(parameters[i]);
            if (biasMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetBiasMode(*biasMode) << "\n";
        } else {
            Gna2PoolingMode* poolingMode = reinterpret_cast<Gna2PoolingMode*>(parameters[i]);
            if (poolingMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetPoolingMode(*poolingMode)
                         << "\n";
        }
        i++;
    }
}

inline void DumpCopyParameters(std::ostream& dumpFile,
                               void** parameters,
                               size_t knownParamCount,
                               const std::vector<std::string> paramNames) {
    Gna2Shape* subTensorShape = reinterpret_cast<Gna2Shape*>(parameters[CopyShapeParamIdx]);
    DumpShape(dumpFile, subTensorShape, paramNames[CopyShapeParamIdx]);
}

inline void DumpFCAffineParameters(std::ostream& dumpFile,
                                   void** parameters,
                                   size_t knownParamCount,
                                   const std::vector<std::string> paramNames) {
    size_t i = 0;

    while (i < knownParamCount) {
        if (i == BiasModeFCAffineParamIdx) {
            Gna2BiasMode* biasMode = reinterpret_cast<Gna2BiasMode*>(parameters[BiasModeFCAffineParamIdx]);
            if (biasMode != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << GetBiasMode(*biasMode) << "\n";
        } else {
            uint32_t* biasVectorIndex = reinterpret_cast<uint32_t*>(parameters[BiasModeFCAffineParamIdx]);
            if (biasVectorIndex != nullptr)
                dumpFile << "\tParameter name: " << paramNames[i] << ", value: " << *biasVectorIndex << "\n";
        }
        i++;
    }
}

inline void DumpIntParameter(std::ostream& dumpFile,
                             void** parameters,
                             size_t knownParamCount,
                             const std::vector<std::string> paramNames) {
    uint32_t* param = reinterpret_cast<uint32_t*>(parameters[0]);
    if (param != nullptr)
        dumpFile << "\tParameter name: " << paramNames[0] << ", value: " << *param << "\n";
}

inline std::vector<std::string> GetParamaterNames(Gna2OperationType type) {
    // This map must be aligned with dumpParamMap in this file
    static const std::map<Gna2OperationType, std::vector<std::string>> operationParamaterNamesMap = {
        {Gna2OperationTypeConvolution,
         {"convolutionStride", "biasMode", "poolingMode", "poolingWindow", "poolingStride", "zeroPadding"}},
        {Gna2OperationTypeCopy, {"shape (sub-tensor shape)"}},
        {Gna2OperationTypeFullyConnectedAffine, {"biasMode", "biasVectorIndex"}},
        {Gna2OperationTypeGmm, {"maximumScore"}},
        {Gna2OperationTypeRecurrent, {"delay"}}};
    return operationParamaterNamesMap.find(type) != operationParamaterNamesMap.end()
               ? operationParamaterNamesMap.find(type)->second
               : std::vector<std::string>{};
}

typedef void (*dumpParameters)(std::ostream&, void**, size_t, const std::vector<std::string>);

inline dumpParameters GetParamDumpFunc(Gna2OperationType type) {
    // This map must be aligned with operationParamaterNamesMap in this file
    static const std::map<Gna2OperationType, dumpParameters> dumpParamMap = {
        {Gna2OperationTypeConvolution, DumpConvolutionParameters},
        {Gna2OperationTypeCopy, DumpCopyParameters},
        {Gna2OperationTypeFullyConnectedAffine, DumpFCAffineParameters},
        {Gna2OperationTypeGmm, DumpIntParameter},
        {Gna2OperationTypeRecurrent, DumpIntParameter}};
    return dumpParamMap.find(type) != dumpParamMap.end() ? dumpParamMap.find(type)->second : nullptr;
}

inline void DumpPwl(std::ostream& dumpFile, const Gna2Tensor& activation) {
    const Gna2PwlSegment* const segments = static_cast<Gna2PwlSegment*>(activation.Data);
    const uint32_t numberOfSegments = activation.Shape.Dimensions[0];

    for (uint32_t k = 0; k < numberOfSegments; k++) {
        uint32_t scale = ((segments[k].xBase & 3) + 1) * 8;
        uint64_t factor = 1ULL << scale;
        int32_t B = segments[k].xBase & 0xfffffffc;
        double a = static_cast<double>(segments[k].Slope) / factor;
        double b = static_cast<double>(segments[k].yBase) - ((static_cast<double>(B) * segments[k].Slope) / factor);

        dumpFile << "\t\tBase input (B) : " << B << ", ";
        dumpFile << "Base output (b) : " << segments[k].yBase << ", ";
        dumpFile << "Slope (S): " << segments[k].Slope << ", ";
        dumpFile << "Shift (scale) : " << scale << ", ";
        dumpFile << "y = (" << a << ")x + (" << b << ")";
        if (segments[k].Slope != 0) {
            double x0 =
                static_cast<double>(B) - ((static_cast<double>(segments[k].yBase) * factor) / segments[k].Slope);
            dumpFile << ", x0 = " << x0;
        }
        dumpFile << "\n";
    }
}

inline void DumpCompoundBias(std::ostream& dumpFile, const Gna2Tensor& tensor) {
    uint32_t i = 0;

    while (i < tensor.Shape.Dimensions[0]) {
        const Gna2CompoundBias* const bias = static_cast<Gna2CompoundBias*>(tensor.Data) + i;
        dumpFile << "\t\tBias for row " << i << " : " << bias->Bias << ", multiplier: " << unsigned(bias->Multiplier)
                 << "\n";
        i++;
    }
}

inline void DumpCharArray(std::ostream& dumpFile, const char* carray, size_t count) {
    size_t i = 0;
    while (*(carray + i) != 0 && i < count) {
        dumpFile << *(carray + i) << " ";
        i++;
    }
    dumpFile << "\n";
}

class GnaMemStats {
public:
    void Add(const std::string memory_tag,
             const std::string operand_name,
             const std::string operation_name,
             const uint32_t size) {
        if (size == 0)
            return;
        const std::pair<std::string, std::string> key = std::make_pair(operation_name, memory_tag + " " + operand_name);
        m_stats[key] = size;
    }
    std::string GetFormattedMemStats() {
        std::stringstream output;
        output << "Per Layer memory statistics: " << std::endl;
        // Assumes that std::map with pair keys is lexicographically ordered which is true in this and previous versions
        // of C++.
        std::string previous_layer = "";
        for (auto pair : m_stats) {
            std::string current_layer = pair.first.first;
            if (current_layer != previous_layer) {
                output << "\t" << current_layer << std::endl;
            }
            output << "\t\t" << pair.first.second << ": " << pair.second << " bytes." << std::endl;
            previous_layer = current_layer;
        }
        output << "Aggregate memory statistics: " << std::endl;
        std::map<std::string, uint32_t> aggr_stats;
        for (auto pair : m_stats) {
            const std::string key = pair.first.second;
            if (aggr_stats.count(key) == 0) {
                aggr_stats[key] = pair.second;
            } else {
                aggr_stats[key] += pair.second;
            }
        }
        for (auto stat : aggr_stats) {
            output << "\t" << stat.first << ": " << stat.second << " bytes." << std::endl;
        }
        return output.str();
    }

private:
    // uint32_t allows for a little above 4 Gb of size without overflow so it's more than sufficient.
    std::map<std::pair<std::string, std::string>, uint32_t> m_stats;
};

void DumpGna2Model(const Gna2Model& gnaModel,
                   const std::string& dumpFolderNameGNA,
                   bool dumpData,
                   const GnaAllocations& allAllocations,
                   const std::string& modeOfOperation) {
    GnaMemStats gna_stats;
    std::stringstream dumpFileName;
    uint32_t opsNo = gnaModel.NumberOfOperations;
    std::time_t currTime = std::time(nullptr);

    dumpFileName << dumpFolderNameGNA << "Gna2ModelDebugDump_" << opsNo << "_layer_"
                 << std::put_time(std::localtime(&currTime), "%Y%m%d%H%M%S") << modeOfOperation;

    std::ofstream dumpFile(dumpFileName.str() + ".txt", std::ios::out);

    const auto& allAllocationsSorted = allAllocations.GetAllocationsInExportOrder();
    for (auto&& a : allAllocationsSorted) {
        dumpFile << "Allocation: ptr=" << a.ptr << "\tsizeRequested=" << a.sizeRequested
                 << "\tsizeGranted=" << a.sizeGranted << "\t tag=" << a.GetTagName() << "\n";
    }

    dumpFile << "Layers (operations) count: " << opsNo << "\n";

    for (size_t i = 0; i < opsNo; i++) {
        const auto& operation = gnaModel.Operations[i];

        dumpFile << "------------------------------------------------------------------------\n\n";

        dumpFile << "Layer (operation): " << i << "\n";
        dumpFile << "Layer (operation) type: " << GetLayerType(operation.Type) << "\n";
        dumpFile << "Number of possible operands: " << operation.NumberOfOperands << "\n";

        for (size_t j = 0; j < operation.NumberOfOperands; j++) {
            if (operation.Operands[j] == nullptr) {
                dumpFile << "\tOperand " << j << " == nullptr\n";
                continue;
            }
            const auto& operand = *operation.Operands[j];
            void* foundPtr = nullptr;
            std::string foundName = "AllocationNotFound";
            size_t offset = 0;
            auto found = std::find_if(allAllocationsSorted.begin(),
                                      allAllocationsSorted.end(),
                                      [operand](const GnaAllocation& allocation) {
                                          return allocation.getOffset(operand.Data).first;
                                      });
            if (found != allAllocationsSorted.end()) {
                foundPtr = found->ptr;
                foundName = found->GetTagName();
                offset = found->getOffset(operand.Data).second;
            }
            const uint32_t size =
                Gna2RoundUp(GetGnaShapeSize(operand.Shape, GetTypeByteSize(operand.Type)),
                            static_cast<uint32_t>(Limitations::get_instance()->get_memory_alignment()));
            dumpFile << "\tOperand " << j << " (" << GetOperandName(operation.Type, j) << ")"
                     << " type: " << GetOperandType(operand.Type) << " shape: " << GetSimpleString(operand.Shape)
                     << " tag: " << foundName << " offset: " << offset << " size: " << size << " data: " << operand.Data
                     << " baseAlloc: " << foundPtr << " layout: ";
            gna_stats.Add(foundName,
                          GetOperandName(operation.Type, j),
                          std::to_string(i) + " " + GetLayerType(operation.Type),
                          size);
            DumpCharArray(dumpFile, operand.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS);

            if (operand.Type == Gna2DataTypePwlSegment) {
                DumpPwl(dumpFile, operand);
            } else if (operand.Type == Gna2DataTypeCompoundBias) {
                DumpCompoundBias(dumpFile, operand);
            } else if (dumpData) {
                std::ofstream datFile(dumpFileName.str() + ".dat", std::ios::app);
                std::vector<uint32_t> elementIndex(operand.Shape.NumberOfDimensions);

                auto beginItr = operand.Shape.Dimensions + 1;
                auto endIter = operand.Shape.Dimensions + operand.Shape.NumberOfDimensions;
                auto columnsValue = std::accumulate(beginItr, endIter, 1, std::multiplies<int>());
                datFile << "Layer " << i << ", type " << GetLayerType(operation.Type) << ", operand " << j << " - "
                        << GetOperandName(operation.Type, j) << ", rows: " << operand.Shape.Dimensions[0]
                        << ", columns: " << columnsValue << "\n";

                uint32_t ind = 0;

                do {
                    int32_t value = GetValue(operand, elementIndex);
                    datFile << std::setw(6) << value;
                    auto postValueCharackters = (++ind % columnsValue == 0) ? "\n" : ", ";
                    datFile << postValueCharackters;
                } while (NextElement(elementIndex, operand.Shape));
            }
        }

        dumpFile << "Parameters: \n";

        if (operation.NumberOfParameters > 0 && GetParamDumpFunc(operation.Type) != nullptr) {
            std::vector<std::string> paramNames = GetParamaterNames(operation.Type);
            if (!paramNames.empty()) {
                size_t knownParamCount = operation.NumberOfParameters <= paramNames.size()
                                             ? operation.NumberOfParameters
                                             : paramNames.size();
                GetParamDumpFunc(operation.Type)(dumpFile, operation.Parameters, knownParamCount, paramNames);
            }
        }
    }
    dumpFile << "------------------------------------------------------------------------\n\n";
    dumpFile << gna_stats.GetFormattedMemStats() << std::endl;
}

}  // namespace dump
}  // namespace intel_gna
}  // namespace ov
