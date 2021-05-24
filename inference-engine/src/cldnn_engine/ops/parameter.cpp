// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/parameter.hpp"

#include "api/input_layout.hpp"
#include "api/reorder.hpp"
#include "api/data.hpp"
#include "api/concatenation.hpp"

using namespace InferenceEngine;

namespace CLDNNPlugin {

void CreateParameterOp(Program& p, const std::shared_ptr<ngraph::op::v0::Parameter>& op) {
    auto networkInputs = p.GetNetworkInputs();
    if (networkInputs.find(op->get_friendly_name()) == networkInputs.end()) {
        IE_THROW() << "Can't find input " << op->get_friendly_name() << " in InputsDataMap";
    }

    auto inputInfo = networkInputs.at(op->get_friendly_name());
    // first create and add the input layout
    const auto inputDesc = inputInfo->getTensorDesc();
    const auto inputDims = inputDesc.getDims();
    Layout l = inputDesc.getLayout();
    Precision ip = inputDesc.getPrecision();

    cldnn::format inputFormat = cldnn::format::bfyx;
    if (Layout::BLOCKED == l && 6 == inputDims.size()) {
        inputFormat = cldnn::format::bfwzyx;
    } else {
        inputFormat = FormatFromLayout(l);
    }

    cldnn::tensor dataTensor;
    cldnn::tensor::value_type batch = (p.m_max_batch <= 1) ? (!inputDims.empty() ? TensorValue(inputDims[0]) : 1)
                                                           : TensorValue(p.m_curBatch);
    switch (inputDims.size()) {
    case 6:
        dataTensor = cldnn::tensor(cldnn::batch(batch),
                                   cldnn::feature(inputDims[1]),
                                   cldnn::spatial(inputDims[5], inputDims[4], inputDims[3], inputDims[2]));
        break;
    case 5:
        if (Layout::NCDHW == l) {
            dataTensor = cldnn::tensor(cldnn::batch(batch),
                                       cldnn::feature(inputDims[1]),
                                       cldnn::spatial(inputDims[4], inputDims[3], inputDims[2]));
        } else {
            IE_THROW()  << "Unsupported layout (" << l << ") in 5D input " << inputInfo->name();
        }
        break;
    case 4:
        if (Layout::NCHW == l || Layout::CHW == l) {
            dataTensor = cldnn::tensor(batch,
                                       TensorValue(inputDims[1]), TensorValue(inputDims[3]), TensorValue(inputDims[2]));
        } else if (Layout::NHWC == l) {
            dataTensor = cldnn::tensor(batch,
                                       TensorValue(inputDims[1]), TensorValue(inputDims[3]), TensorValue(inputDims[2]));
        } else {
            IE_THROW() << "Unsupported layout (" << l << ") in 4D input " + inputInfo->name();
        }
        break;
    case 3:
        if (Layout::CHW == l) {
            dataTensor = cldnn::tensor(TensorValue(inputDims[0]), TensorValue(inputDims[1]), 1, TensorValue(inputDims[2]));
        } else {
            IE_THROW() << "Unsupported layout (" << l << ") in 3D input " + inputInfo->name();
        }
        break;
    case 2:
        if (Layout::NCHW == l || NC == l) {
            dataTensor = cldnn::tensor(batch, TensorValue(inputDims[1]), 1, 1);
        } else {
            IE_THROW() << "Unsupported layout (" << l << ") in 2D input " << inputInfo->name();
        }
        break;
    case 1:
        dataTensor = cldnn::tensor(TensorValue(inputDims[0]), 1, 1, 1);
        break;
    case 0:
        dataTensor = cldnn::tensor(1, 1, 1, 1);
        break;
    default: IE_THROW() << "Invalid data dimensions";
    }
    cldnn::layout networkInputLayout(DataTypeFromPrecision(ip),
                                     inputFormat,
                                     dataTensor);

    // look at the expected color format of this input
    auto inputName = layer_type_name_ID(op);
    auto preProcess = inputInfo->getPreProcess();
    size_t meanChannels = preProcess.getNumberOfChannels();
    networkInputLayout.format = inputFormat;
    networkInputLayout.size = networkInputLayout.size.transform(inputFormat, 1);
    networkInputLayout.data_type = DataTypeFromPrecision(op->get_output_element_type(0));
    cldnn::primitive_id meanBlobID = inputName + Program::m_meanValuesTag;
    std::vector<float> meanValues;

    if ((meanChannels > 0) &&
        (meanChannels != networkInputLayout.size.feature[0])) {
        IE_THROW() << "Mismatched mean values channels in input " << inputName;
    }

    switch (preProcess.getMeanVariant()) {
    case NONE:
    case MEAN_VALUE: {
        if (meanChannels > 0) {
            for (size_t c = 0; c < meanChannels; c++) {
                if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                    IE_THROW() << "not supporting stdScale yet in input " << inputName;
                meanValues.push_back(preProcess[c]->meanValue);
            }
        }
        break;
    }
    case MEAN_IMAGE: {
        IE_ASSERT(meanChannels);
        // first merge all mean values to a single blob
        // todo make sure mean blob precision is the same as the input precision
        auto meanDims = inputDims;
        // overwrite batches with 1
        switch (meanDims.size()) {
        case 4: meanDims[0] = 1;
            break;
        default:
            IE_THROW() << "Missing batch dimensions in input image";
        }
        const TensorDesc desc(Precision::FP32, meanDims, TensorDesc::getLayoutByDims(meanDims));
        TBlob<float> meanBlob(desc);
        meanBlob.allocate();
        auto meanBlobData = meanBlob.data();
        for (size_t c = 0; c < meanChannels; c++) {
            if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                IE_THROW() << "not supporting stdScale yet in input " << inputName;
            auto channelMeanBlob = std::dynamic_pointer_cast<TBlob<float>>(preProcess[c]->meanData);
            auto channelSize = channelMeanBlob->size();
            auto channelBlobData = channelMeanBlob->data();
            for (size_t i = 0; i < channelSize; i++) {
                meanBlobData[(c * channelSize) + i] = channelBlobData[i];
            }
        }
        // then create a data primitive for the mean values
        auto meanBlobPtr = std::make_shared<TBlob<float>>(meanBlob);

        // mean values will use external format (sub in the input format before convert to new format)
        cldnn::tensor meanBlobTensor(networkInputLayout.size);
        meanBlobTensor.batch[0] = 1;  // mean values have no batches
        cldnn::layout meanBlobLayout(cldnn::data_types::f32, cldnn::format::bfyx, meanBlobTensor);

        auto data = static_cast<const char *>(meanBlobPtr->buffer());

        auto bufIter = p.blobMemCache.find(std::make_pair(data, meanDims));
        if (bufIter != p.blobMemCache.end()) {
            meanBlobID = bufIter->second;
        } else {
            auto mem = cldnn::memory::allocate(p.GetEngine(), meanBlobLayout, 0, false);
            auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
            auto buf = tmpPointer.data();
            auto bufSize = meanBlobLayout.bytes_count();

            std::memcpy(&buf[0], &data[0], bufSize);

            p.AddPrimitive(cldnn::data(meanBlobID, mem));
            p.blobMemCache[std::make_pair(data, meanDims)] = meanBlobID;
        }
        break;
    }
    default: IE_THROW() << "Invalid mean variant in input " << inputName;
        break;
    }

    if (ColorFormat::NV12 == preProcess.getColorFormat() && p.GetConfig().nv12_two_inputs) {
        // for NV12, create two input layouts with reorder instead of one,
        // and then would expect compound blob in inferRequest
        if (Layout::NCHW != l &&
            (Precision::I8 != ip || Precision::U8 != ip)) {
            IE_THROW() << "Unsupported layout (" << l << ") or precision "
                               << ip.name() << ") for NV12 input " + inputInfo->name();
        }
        int height = inputDims[2];
        int width = inputDims[3];
        std::vector<cldnn::primitive_id> reorders;

        for (auto i = 0; i < inputDims[0]; i++) {
            auto preprocessPrimID = "reorder:" + inputName + std::to_string(i) + Program::m_preProcessTag;
            std::string y_name = inputName + "_Y" + std::to_string(i);
            std::string uv_name = inputName + "_UV" + std::to_string(i);

            cldnn::layout y_layout(DataTypeFromPrecision(ip),
                                    cldnn::format::nv12, { 1, 1, width, height });
            cldnn::layout uv_layout(DataTypeFromPrecision(ip),
                                    cldnn::format::nv12, { 1, 2, width / 2, height / 2 });
            auto inputY = cldnn::input_layout(y_name, y_layout);
            auto inputUV = cldnn::input_layout(uv_name, uv_layout);

            p.AddPrimitive(inputY);
            p.inputLayouts.insert({ inputInfo->name() + "_Y" + std::to_string(i), y_layout });
            p.AddPrimitive(inputUV);
            p.inputLayouts.insert({ inputInfo->name() + "_UV" + std::to_string(i), uv_layout });
            switch (preProcess.getMeanVariant()) {
            case NONE:
            case MEAN_VALUE: {
                p.AddPrimitive(cldnn::reorder(preprocessPrimID, y_name, uv_name, networkInputLayout, meanValues));
                break;
            }
            case MEAN_IMAGE: {
                p.AddPrimitive(cldnn::reorder(preprocessPrimID, y_name, uv_name, networkInputLayout, meanBlobID));
                break;
            }
            default: IE_THROW(Unexpected) << "Invalid mean variant in input " + inputName;
                break;
            }

            p.primitivesToIRLayersMap[preprocessPrimID] = { inputInfo->name() };
            p.primitivesToIRLayersMap[y_name] = { inputInfo->name() };
            p.primitivesToIRLayersMap[uv_name] = { inputInfo->name() };
            p.profilingIDs.push_back(preprocessPrimID);
            p.InitProfileInfo(preprocessPrimID, "Reorder");
            p.primitiveIDs[inputName] = preprocessPrimID;  // If it is batched blob, it will be overwritten afterwards.
            p.primitiveIDs[preprocessPrimID] = preprocessPrimID;
            reorders.push_back(preprocessPrimID);
        }

        if (inputDims[0] > 1) {
            auto concatPrimID = "concat:" + inputName + Program::m_preProcessTag;
            p.AddPrimitive(cldnn::concatenation(concatPrimID, reorders, cldnn::concatenation::along_b));
            p.primitiveIDs[inputName] = concatPrimID;
        }
    } else {
        auto preprocessPrimID = "reorder:" + inputName + Program::m_preProcessTag;
        cldnn::layout inputLayout(networkInputLayout);
        inputLayout.data_type = DataTypeFromPrecision(ip);
        p.inputLayouts.insert({ inputInfo->name(), inputLayout });

        p.AddPrimitive(cldnn::input_layout(inputName, inputLayout));
        p.primitivesToIRLayersMap[inputName] = { inputInfo->name() };

        switch (preProcess.getMeanVariant()) {
        case NONE:
        case MEAN_VALUE: {
            p.AddPrimitive(cldnn::reorder(preprocessPrimID, inputName, networkInputLayout, meanValues));
            break;
        }
        case MEAN_IMAGE: {
            p.AddPrimitive(cldnn::reorder(preprocessPrimID,
                                        inputName,
                                        networkInputLayout,
                                        meanBlobID));
            break;
        }
        default: IE_THROW() << "Invalid mean variant in input " << inputName;
            break;
        }
        p.InitProfileInfo(preprocessPrimID, "reorder");
        p.primitiveIDs[preprocessPrimID] = preprocessPrimID;
        p.primitiveIDs[inputName] = preprocessPrimID;
        p.profilingIDs.push_back(preprocessPrimID);
    }
}

REGISTER_FACTORY_IMPL(v0, Parameter);

}  // namespace CLDNNPlugin
