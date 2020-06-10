// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna2_model_helper.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#if GNA_LIB_VER == 2
#include "gna2_model_debug_log.hpp"
#include "gna2-model-api.h"
#include <details/ie_exception.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <sstream>
#include <string>
#include <cmath>

std::string getLayerType(Gna2OperationType kind) {
    switch (kind) {
    case Gna2OperationTypeFullyConnectedAffine: return "Gna2OperationTypeFullyConnectedAffine";
    case Gna2OperationTypeElementWiseAffine: return "Gna2OperationTypeElementWiseAffine";
    case Gna2OperationTypeRecurrent: return "Gna2OperationTypeRecurrent";
    case Gna2OperationTypeConvolution: return "Gna2OperationTypeConvolution";
    case Gna2OperationTypeTransposition: return "Gna2OperationTypeTransposition";
    case Gna2OperationTypeCopy: return "Gna2OperationTypeCopy";
    default: return "Gna2OperationUNKNOWN";
    }
}

void write_pwl(std::ostream & pwl_file, const Gna2PwlSegment* const segments, const uint32_t numberOfSegments) {
    for (uint32_t k = 0; k < numberOfSegments; k++) {
        pwl_file << segments[k].Slope << ", " << segments[k].xBase << ", " << segments[k].yBase << "\n";
    }
}

void write_pwl(std::ostream & pwl_file, const Gna2Tensor& activation) {
    write_pwl(pwl_file, static_cast<Gna2PwlSegment*>(activation.Data), activation.Shape.Dimensions[0]);
}

std::string GetSimpleString(Gna2Shape shape) {
    std::stringstream out;
    for (uint32_t i = 0; i < shape.NumberOfDimensions; i++) {
        out << shape.Dimensions[i];
        if (i + 1 < shape.NumberOfDimensions) out << 'x';
    }
    return out.str();
}
template <class T>
bool NextElement(T & elementIndex, const Gna2Shape& total) {
    if (total.NumberOfDimensions == 0) return false;
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
uint32_t GetLinearIndex(const T & elementIndex, const Gna2Shape& total) {
    uint32_t out = 0;
    for (int idx = 0; idx < total.NumberOfDimensions; idx++) {
        out += elementIndex[idx];
        if (idx + 1 < total.NumberOfDimensions) out *= total.Dimensions[idx + 1];
    }
    return out;
}

template <class T>
int32_t GetValue(const Gna2Tensor& tensor, const T & elementIndex) {
    int32_t intValue = -123;  // fake value indicating problem
    const auto linearIndex = GetLinearIndex(elementIndex, tensor.Shape);
    if (tensor.Type == Gna2DataTypeInt32) {
        intValue = (reinterpret_cast<int32_t *>(tensor.Data)[linearIndex]);
    } else if (tensor.Type == Gna2DataTypeInt16) {
        intValue = reinterpret_cast<int16_t *>(tensor.Data)[linearIndex];
    }
    return intValue;
}

void WriteInputAndOutputTextGNAImpl(const Gna2Model & gnaModel, const std::string dumpFolderNameGNA, const std::string refFolderName) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        const auto & operation = gnaModel.Operations[i];
        std::stringstream out_file_name;
        const auto & outputTensor = *operation.Operands[OutOpIdx];
        const auto & intputTensor = *operation.Operands[InOpIdx];

        out_file_name << std::setfill('0') << std::setw(2) << i << "_"
            << getLayerType(operation.Type)
            << "-" << GetSimpleString(intputTensor.Shape)
            << "-" << GetSimpleString(outputTensor.Shape);

        auto inputfileName = dumpFolderNameGNA + out_file_name.str() + "_input.txt";
        auto outFileName = dumpFolderNameGNA + out_file_name.str() + "_output.txt";
        auto pwlFileName = dumpFolderNameGNA + out_file_name.str() + "_pwl.txt";
        auto refOutputFileName = refFolderName + out_file_name.str() + "_output.txt";

        std::ofstream out_file(outFileName.c_str(), std::ios::out);
        std::ofstream pwl_file(pwlFileName.c_str(), std::ios::out);
        std::ifstream ref_out_file(refOutputFileName.c_str(), std::ios::in);
        std::ofstream in_file(inputfileName.c_str(), std::ios::out);

        float  summOfDiff = 0.f;
        float  summOfSqDiff = 0.f;
        float  maxD = 0.0f;
        int    numItems = 0;

        if (operation.NumberOfOperands > PwlOpIdx && operation.Operands[PwlOpIdx] != nullptr) {
            write_pwl(pwl_file, *operation.Operands[PwlOpIdx]);
        }

        std::vector<uint32_t> elementIndex(outputTensor.Shape.NumberOfDimensions);

        do {
            float floatValue = GetValue(outputTensor, elementIndex);

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
            std::cout << std::left << std::setw(55) << out_file_name.str()
                << " RMSE=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << rmse
                << " avg=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << avg
                << " maxD=" << std::fixed << std::setprecision(5) << std::right << std::setw(8) << maxD << std::endl;
        }

        std::vector<uint32_t> inputElementIndex(intputTensor.Shape.NumberOfDimensions);

        do {
            int32_t intValue = GetValue(intputTensor, inputElementIndex);
            in_file << std::setw(8) << intValue;
            in_file << "\n";
        } while (NextElement(inputElementIndex, intputTensor.Shape));
    }
}

#endif
