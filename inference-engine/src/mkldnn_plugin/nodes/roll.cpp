// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <string>
#include <vector>
#include <cmath>

#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include "mkldnn/ie_mkldnn.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class RollImpl: public ExtLayerBase {
public:
    explicit RollImpl(const CNNLayer* layer) {
        try {
            layerName = layer->name;
            if (layer->insData.size() != numberOfInputs) {
                IE_THROW() << "Roll layer with name '" << layerName << "' has incorrect number of input/output edges!";
            }

            /* Data */
            const auto &dataTensor = layer->insData[DATA_INDEX].lock()->getTensorDesc();
            const auto &dataShape = dataTensor.getDims();
            if (dataShape.size() < 1) {
                IE_THROW() << "Roll layer with name '" << layerName << "' doesn't support 'data' input tensor with rank: " << dataShape.size();
            }
            m_numOfDims = dataShape.size();

            const auto& dataPrecision = dataTensor.getPrecision();
            if (dataPrecision != Precision::I8 && dataPrecision != Precision::U8 && dataPrecision != Precision::I16 && dataPrecision != Precision::I32 &&
                dataPrecision != Precision::FP32 && dataPrecision != Precision::I64 && dataPrecision != Precision::U64 && dataPrecision != Precision::BF16) {
                IE_THROW() << "Roll layer with name '" << layerName << "' has unsupported 'data' input precision: " << dataPrecision.name();
            }

            if (dataShape != layer->outData[0]->getTensorDesc().getDims()) {
                IE_THROW() << "Roll layer with name '" << layerName << "' has different 'data' input and output dimensions";
            }

            /* Axes */
            const auto& axesTensor = layer->insData[AXES_INDEX].lock()->getTensorDesc();
            const auto& axesTensorPrec = layer->insData[AXES_INDEX].lock()->getTensorDesc().getPrecision();
            if (axesTensorPrec != Precision::I32 && axesTensorPrec != Precision::I64) {
                IE_THROW() << "Roll layer with name '" << layerName << "' has unsupported 'axes' input precision: " << axesTensorPrec.name();
            }

            const auto axesTensorRank = axesTensor.getDims().size();
            if (axesTensorRank > 1) {
                IE_THROW() << "Roll layer with name '" << layerName << "' doesn't support 'axes' input tensor with rank: " << axesTensorRank;
            }

            /* Shift */
            const auto& shiftTensor = layer->insData[SHIFT_INDEX].lock()->getTensorDesc();
            const auto& shiftTensorPrec = layer->insData[SHIFT_INDEX].lock()->getTensorDesc().getPrecision();
            if (shiftTensorPrec != Precision::I32 && shiftTensorPrec != Precision::I64) {
                IE_THROW() << "Roll layer with name '" << layerName << "' has unsupported 'shift' input precision: " << shiftTensorPrec.name();
            }

            const auto shiftTensorRank = shiftTensor.getDims().size();
            if (shiftTensorRank > 1) {
                IE_THROW() << "Roll layer with name '" << layerName << "' doesn't support 'shift' input tensor with rank: " << shiftTensorRank;
            }

            m_shape = dataShape;

            LayerConfig config;
            for (size_t i = 0; i < layer->insData.size(); i++) {
                DataConfig inConfig;
                inConfig.inPlace = -1;
                inConfig.constant = false;

                Precision inPrecision = i > DATA_INDEX ? Precision(Precision::I32) : layer->insData[i].lock()->getTensorDesc().getPrecision();
                if (inPrecision == Precision::BF16)
                    inPrecision = Precision::FP32;
                const SizeVector& inDims = layer->insData[i].lock()->getTensorDesc().getDims();
                inConfig.desc = TensorDesc(inPrecision, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));
                config.inConfs.push_back(inConfig);
            }
            DataConfig outConfig;
            outConfig.inPlace = -1;
            outConfig.constant = false;
            Precision outPrecision = layer->insData[DATA_INDEX].lock()->getTensorDesc().getPrecision();
            if (outPrecision == Precision::BF16) {
                outPrecision = Precision::FP32;
            }
            const SizeVector& outDims = layer->outData[0]->getTensorDesc().getDims();
            outConfig.desc = TensorDesc(outPrecision, outDims, InferenceEngine::TensorDesc::getLayoutByDims(outDims));

            config.outConfs.push_back(outConfig);

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs, ResponseDesc *resp) noexcept {
        m_shape = inputs[DATA_INDEX]->getTensorDesc().getDims();
        m_numOfDims = m_shape.size();
        m_shifts = extract1dIntegerDataFromBlob(inputs[SHIFT_INDEX]);
        m_axes = extract1dIntegerDataFromBlob(inputs[AXES_INDEX]);

        const auto &dataPrecision = inputs[DATA_INDEX]->getTensorDesc().getPrecision();
        switch (dataPrecision) {
            case Precision::I8: {
                rollImpl<int8_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::U8: {
                rollImpl<uint8_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::I16: {
                rollImpl<int16_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::I32: {
                rollImpl<int32_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::FP32: {
                rollImpl<float>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::I64: {
                rollImpl<int64_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            case Precision::U64: {
                rollImpl<uint64_t>(inputs[DATA_INDEX], outputs[0]);
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "Roll layer with name has unsupported 'data' input precision: ";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
        return OK;
    }

private:
    struct AxisShift;
    std::vector<int64_t> extract1dIntegerDataFromBlob(const Blob::CPtr &data) {
        const auto &dataPrecision = data->getTensorDesc().getPrecision();
        const auto &dataDims = data->getTensorDesc().getDims();
        size_t nElements = 1;
        std::vector<int64_t> dataFromBlob;
        if (!dataDims.empty()) {
            dataFromBlob.reserve(dataDims[0]);
            nElements = dataDims[0];
        }
        size_t padding = data->getTensorDesc().getBlockingDesc().getOffsetPadding();
        switch (dataPrecision) {
            case Precision::I32: {
                const auto *dataPtr = data->cbuffer().as<const int32_t *>() + padding;
                for (size_t idx = 0; idx < nElements; ++idx) {
                    dataFromBlob.push_back(dataPtr[idx]);
                }
                break;
            }
            case Precision::I64: {
                const auto *dataPtr = data->cbuffer().as<const int64_t *>() + padding;
                for (size_t idx = 0; idx < nElements; ++idx) {
                    dataFromBlob.push_back(dataPtr[idx]);
                }
                break;
            }
        }
        return dataFromBlob;
    }

    void parallelItInit(size_t start, std::vector<size_t> &counters, const std::vector<size_t> &iterationRange) {
        auto itCounter = counters.rbegin();
        auto itWork = iterationRange.rbegin();
        while (itCounter != counters.rend()) {
            *itCounter = start % *itWork;
            start /= *itWork;
            ++itCounter;
            ++itWork;
        }
    }

    inline void parallelItStep(std::vector<size_t> &counters, const std::vector<size_t> &iterationRange) {
        auto itCounter = counters.rbegin();
        auto itWork = iterationRange.rbegin();

        while (itCounter != counters.rend()) {
            *itCounter = (*itCounter + 1) % *itWork;
            if (*itCounter != 0) {
                break;
            }
            ++itCounter;
            ++itWork;
        }
    }

    // returns a vector sorted in ascending order by axes
    inline std::vector<AxisShift> calculateShifts(const std::vector<int64_t> &axes, const std::vector<int64_t> &shifts) {
        std::map<size_t, size_t> axesToShift;
        const int64_t numberOfDims = static_cast<int64_t>(m_shape.size());
        for (size_t index = 0; index < axes.size(); ++index) {
            int64_t axis = axes[index];

            if (axis >= numberOfDims || axis < -numberOfDims) {
                IE_THROW() << "Axis: " << axis << " is out of bounds for Roll layer with name: " << layerName;
            }

            if (axis < 0) {
                axis += numberOfDims;
            }

            int64_t shift = shifts[index];
            const int64_t currentDim = static_cast<int64_t>(m_shape[axis]);

            if (shift >= currentDim || shift < -currentDim) {
                IE_THROW() << "Shift: " << shift << " " << currentDim << " is out of bounds for Roll layer with name: " << layerName;
            }

            if (shift < 0) {
                shift += currentDim;
            }
            axesToShift[static_cast<size_t>(axis)] += static_cast<size_t>(shift);
        }
        std::vector<AxisShift> result(axesToShift.size());
        size_t resultIndex = 0;
        // We have array sorted by axes
        for (auto it = axesToShift.begin(); it != axesToShift.end(); ++it) {
            result[resultIndex++] = {it->first, it->second};
        }
        return result;
    }

    template<typename T>
    inline T *getElementByIndex(T *data, const std::vector<size_t> &counters, const std::vector<size_t> &strides) {
        size_t offset = 0;
        for (size_t idx = 0; idx < counters.size(); ++idx) {
            offset += counters[idx] * strides[idx];
        }
        return data + offset;
    }

    template<typename T>
    inline T *getElementByIndexWithShifts(T *data, const std::vector<size_t> &counters, const std::vector<size_t> &strides,
                                          const std::vector<AxisShift> &axesShifts) {
        size_t offset = 0;
        auto it = axesShifts.begin();
        for (size_t idx = 0; idx < counters.size(); ++idx) {
            size_t current_axis_value = counters[idx];
            if (it->axis == idx) {
                current_axis_value += it->shift;
                current_axis_value %= m_shape[idx];
                ++it;
            }
            offset += current_axis_value * strides[idx];
        }
        return data + offset;
    }

    template<typename DataType>
    void rollImpl(const Blob::CPtr &_input, const Blob::Ptr &_output) {
        std::vector<AxisShift> axisShifts = calculateShifts(m_axes, m_shifts);

        const auto *input =
                _input->cbuffer().as<const DataType *>() + _input->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *output = _output->buffer().as<DataType *>() + _output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const std::vector<size_t> strides = _input->getTensorDesc().getBlockingDesc().getStrides();
        size_t workAmount = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
        const size_t cache_line_per_tensor = std::max(1.0f, std::floor(static_cast<float>(workAmount) / mkldnn::utils::get_cache_size(3, false)));
        int threads_number = std::min(parallel_get_max_threads(), static_cast<int>(cache_line_per_tensor));
        parallel_nt(threads_number, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            SizeVector counters(m_numOfDims, 0);
            splitter(workAmount, nthr, ithr, start, end);
            parallelItInit(start, counters, m_shape);
            for (size_t iwork = start; iwork < end; ++iwork) {
                DataType *elementPtrOutput = getElementByIndexWithShifts(output, counters, strides, axisShifts);
                const DataType *elementPtrInput = getElementByIndex(input, counters, strides);
                *elementPtrOutput = *elementPtrInput;

                parallelItStep(counters, m_shape);
            }
        });
    }

    const size_t DATA_INDEX = 0ul;
    const size_t SHIFT_INDEX = 1ul;
    const size_t AXES_INDEX = 2ul;
    const size_t numberOfInputs = 3ul;

    std::vector<int64_t> m_shifts;
    std::vector<int64_t> m_axes;
    size_t m_numOfDims;
    std::vector<size_t> m_shape;

    struct AxisShift {
        size_t axis;
        size_t shift;
    };

    std::string layerName;
};

REG_FACTORY_FOR(RollImpl, Roll);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
