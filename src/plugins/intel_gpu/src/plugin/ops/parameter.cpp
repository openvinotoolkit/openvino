// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/concatenation.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_gpu {

static void CreateParameterOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Parameter>& op) {
    auto networkInputs = p.GetNetworkInputs();
    OPENVINO_ASSERT(networkInputs.find(op->get_friendly_name()) != networkInputs.end(),
                    "[GPU] Can't find input ", op->get_friendly_name(), " in InputsDataMap");

    auto inputInfo = networkInputs.at(op->get_friendly_name());
    // first create and add the input layout
    const auto inputDesc = inputInfo->getTensorDesc();
    InferenceEngine::Layout l = inputDesc.getLayout();
    InferenceEngine::Precision ip = inputDesc.getPrecision();

    auto input_pshape = op->get_partial_shape();
    if (!p.use_new_shape_infer()) {
        if (input_pshape.size() < 4) {
            input_pshape.insert(input_pshape.end(), 4 - input_pshape.size(), ov::Dimension(1));
        }
        if (p.m_max_batch > 1) {
            input_pshape[0] = ov::Dimension(p.m_curBatch);
        }
    }

    cldnn::format inputFormat = cldnn::format::get_default_format(input_pshape.size());
    std::vector<size_t> default_order(input_pshape.size());
    std::iota(default_order.begin(), default_order.end(), 0);
    // For legacy API we need to handle NHWC as well, so check non default order
    if (inputDesc.getBlockingDesc().getOrder() != default_order) {
        inputFormat = FormatFromLayout(l);
    }

    // look at the expected color format of this input
    auto inputName = layer_type_name_ID(op);
    auto preProcess = inputInfo->getPreProcess();
    size_t meanChannels = preProcess.getNumberOfChannels();
    cldnn::layout networkInputLayout(input_pshape,
                                     cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                     inputFormat);
    cldnn::primitive_id meanBlobID = inputName + ProgramBuilder::m_meanValuesTag;
    std::vector<float> meanValues;

    if ((meanChannels > 0) &&
        (meanChannels != static_cast<size_t>(networkInputLayout.feature()))) {
        OPENVINO_THROW("Mismatched mean values channels in input ", inputName);
    }

    switch (preProcess.getMeanVariant()) {
    case NONE:
    case MEAN_VALUE: {
        if (meanChannels > 0) {
            for (size_t c = 0; c < meanChannels; c++) {
                if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                    OPENVINO_THROW("not supporting stdScale yet in input ", inputName);
                meanValues.push_back(preProcess[c]->meanValue);
            }
        }
        break;
    }
    case MEAN_IMAGE: {
        OPENVINO_ASSERT(meanChannels);
        // first merge all mean values to a single blob
        // todo make sure mean blob precision is the same as the input precision
        auto meanDims = input_pshape;
        // overwrite batches with 1
        switch (meanDims.size()) {
        case 4: meanDims[0] = 1;
            break;
        default:
            OPENVINO_THROW("Missing batch dimensions in input image");
        }
        const TensorDesc desc(Precision::FP32, meanDims.to_shape(), TensorDesc::getLayoutByDims(meanDims.to_shape()));
        TBlob<float> meanBlob(desc);
        meanBlob.allocate();
        auto meanBlobData = meanBlob.data();
        for (size_t c = 0; c < meanChannels; c++) {
            if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                OPENVINO_THROW("not supporting stdScale yet in input ", inputName);
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
        cldnn::tensor meanBlobTensor(networkInputLayout.get_tensor());
        meanBlobTensor.batch[0] = 1;  // mean values have no batches
        cldnn::layout meanBlobLayout(cldnn::data_types::f32, cldnn::format::bfyx, meanBlobTensor);

        auto data = static_cast<const char *>(meanBlobPtr->buffer());

        auto bufIter = p.blobMemCache.find(std::make_pair(data, meanDims.to_shape()));
        if (bufIter != p.blobMemCache.end()) {
            meanBlobID = bufIter->second;
        } else {
            auto mem = p.get_engine().allocate_memory(meanBlobLayout, false);
            cldnn::mem_lock<int8_t> tmpPointer{ mem, p.get_engine().get_service_stream() };
            auto buf = tmpPointer.data();
            auto bufSize = meanBlobLayout.bytes_count();

            std::memcpy(&buf[0], &data[0], bufSize);

            p.add_primitive(*op, cldnn::data(meanBlobID, mem));
            p.blobMemCache[std::make_pair(data, meanDims.to_shape())] = meanBlobID;
        }
        break;
    }
    default: OPENVINO_THROW("Invalid mean variant in input ", inputName);
        break;
    }

    auto is_convert_color_type = [](const std::shared_ptr<ov::Node> &node) {
        return ov::is_type<ov::op::v8::NV12toRGB>(node) ||
               ov::is_type<ov::op::v8::NV12toBGR>(node) ||
               ov::is_type<ov::op::v8::I420toRGB>(node) ||
               ov::is_type<ov::op::v8::I420toBGR>(node);
    };

    std::function<bool(const std::shared_ptr<ov::Node>&, size_t)> recursive_search_convert_color =
        [&](const std::shared_ptr<ov::Node> &node, size_t curr_depth) -> bool {
        bool convert_color_found = is_convert_color_type(node);
        if (curr_depth != 0) {
            for (auto& user : node->get_users()) {
                convert_color_found |= recursive_search_convert_color(user, curr_depth - 1);
            }
        }
        return convert_color_found;
    };

    std::function<bool(const std::shared_ptr<ov::Node>&)> has_surface_input =
        [](const std::shared_ptr<ov::Node> &node) -> bool {
        bool surface_input_found = false;
        if (node->output(0).get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
            std::string mem_type = node->output(0).get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                                .as<ov::preprocess::TensorInfoMemoryType>().value;
            if (mem_type.find(ov::intel_gpu::memory_type::surface) != std::string::npos) {
                surface_input_found = true;
            }
        }
        return surface_input_found;
    };

    std::function<bool(const std::shared_ptr<ov::Node>&)> connected_to_quantize =
        [&](const std::shared_ptr<ov::Node> &node) -> bool {
        for (auto& user : node->get_users()) {
            if (ov::is_type<ov::op::v0::FakeQuantize>(user))
                return true;
        }
        return false;
    };

    size_t search_depth = 3;
    bool is_convert_color_input = recursive_search_convert_color(op, search_depth);
    bool is_surface_input = has_surface_input(op);

    if (is_surface_input) {
        size_t batch = input_pshape[0].get_length();
        networkInputLayout.format = cldnn::format::nv12;
        networkInputLayout.set_partial_shape({ 1, input_pshape[3], input_pshape[1], input_pshape[2] });

        std::string suffix = "";
        std::vector<cldnn::input_info> surfaces_inputs;
        for (size_t i = 0; i < batch; ++i) {
            if (batch > 1)
                suffix = "_" + std::to_string(i);
            std::string batched_name = inputName + suffix;
            p.inputLayouts.insert({ inputInfo->name() + suffix, networkInputLayout });
            p.add_primitive(*op, cldnn::input_layout(batched_name, networkInputLayout));

            auto reorder_layout = networkInputLayout;
            reorder_layout.format = cldnn::format::bfyx;

            auto preprocessPrimID = "reorder:" + inputName + ProgramBuilder::m_preProcessTag + suffix;
            auto reorder = cldnn::reorder(preprocessPrimID,
                                          cldnn::input_info(batched_name),
                                          reorder_layout);
            reorder.input_mem_type = cldnn::reorder::memory_type::surface;
            p.add_primitive(*op, reorder);
            surfaces_inputs.push_back(cldnn::input_info(preprocessPrimID));
        }

        if (batch > 1 && !is_convert_color_input)
            p.add_primitive(*op, cldnn::concatenation(inputName, surfaces_inputs, 0));
        else
            p.primitive_ids[inputName] = "reorder:" + inputName + ProgramBuilder::m_preProcessTag;
    } else if (is_convert_color_input) {
        networkInputLayout.format = cldnn::format::byxf;

        networkInputLayout.set_partial_shape({ input_pshape[0], input_pshape[3], input_pshape[1], input_pshape[2] });

        p.inputLayouts.insert({ inputInfo->name(), networkInputLayout });
        p.add_primitive(*op, cldnn::input_layout(inputName, networkInputLayout));
    } else {
        auto preprocessPrimID = "reorder:" + inputName + ProgramBuilder::m_preProcessTag;
        cldnn::layout inputLayout(networkInputLayout);
        auto network_input_data_type = DataTypeFromPrecision(ip);
        inputLayout.data_type = network_input_data_type;
        p.inputLayouts.insert({ inputInfo->name(), inputLayout });

        p.add_primitive(*op, cldnn::input_layout(inputName, inputLayout));

        switch (preProcess.getMeanVariant()) {
        case NONE: {
            // If mean value is not specified and the data type does not change, do not add post reorder
            if (network_input_data_type != networkInputLayout.data_type || connected_to_quantize(op)) {
                p.add_primitive(*op, cldnn::reorder(preprocessPrimID,
                                                    cldnn::input_info(inputName),
                                                    networkInputLayout,
                                                    meanValues,
                                                    cldnn::reorder_mean_mode::none), {inputName});
            }
            break;
        }
        case MEAN_VALUE: {
            p.add_primitive(*op, cldnn::reorder(preprocessPrimID,
                                                cldnn::input_info(inputName),
                                                networkInputLayout,
                                                meanValues,
                                                cldnn::reorder_mean_mode::subtract), {inputName});
            break;
        }
        case MEAN_IMAGE: {
            p.add_primitive(*op, cldnn::reorder(preprocessPrimID,
                                                cldnn::input_info(inputName),
                                                networkInputLayout,
                                                meanBlobID,
                                                cldnn::reorder_mean_mode::subtract), {inputName});
            break;
        }
        default: OPENVINO_THROW("Invalid mean variant in input ", inputName);
            break;
        }
    }
}

REGISTER_FACTORY_IMPL(v0, Parameter);

}  // namespace intel_gpu
}  // namespace ov
