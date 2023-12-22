// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network_reader.hpp"

#include <fstream>
#include <istream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_cnn_network.h"
#include "dev/converter_utils.hpp"
#include "file_utils.h"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_input_info.hpp"
#include "itt.hpp"
#include "legacy_op_extension.hpp"
#include "ngraph/function.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/shared_object.hpp"
#include "so_ptr.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

namespace {

CNNNetwork convert_to_cnnnetwork(std::shared_ptr<ngraph::Function>& function,
                                 const std::vector<IExtensionPtr>& exts,
                                 bool is_new_api,
                                 bool frontendMode = false) {
    // only for IR cases we need preprocessing or postprocessing steps
    if (function->has_rt_info("version") && function->get_rt_info<int64_t>("version") == 11 && !is_new_api) {
        IR_READER_SCOPE(ir11_old_api);
        ov::preprocess::PrePostProcessor prepost(function);

        const std::string& old_api_map_key_order = ov::OldApiMapOrder::get_type_info_static();
        const std::string& old_api_map_key_type = ov::OldApiMapElementType::get_type_info_static();

        bool need_validate_nodes_and_infer_types = false;
        auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto& parameter = parameters[i];
            ov::RTMap& rtInfo = parameter->get_rt_info();
            const auto it_type = rtInfo.find(old_api_map_key_type);
            auto& pre_input = prepost.input(i);
            if (it_type != rtInfo.end()) {
                const auto old_api_map_type = it_type->second.as<ov::OldApiMapElementType>().value;
                const auto param_type = parameter->get_element_type();

                // In the following code we add Convert node from old_api_map_type to Parameter type
                // using PrePostProcessor. As some plugins do not support uint8 type, Convert to uint8 leads
                // to error, so for such case type is set directly to Parameter node instead of inserting Convert.
                if ((param_type == ngraph::element::u8 && old_api_map_type.is_real())) {
                    parameter->set_element_type(old_api_map_type);
                    need_validate_nodes_and_infer_types = true;
                } else {
                    pre_input.tensor().set_element_type(old_api_map_type);
                }

                OPENVINO_ASSERT(!old_api_map_type.is_dynamic(), "Old API map does not support dynamic type");
                rtInfo.erase(it_type);
            }
            const auto it_order = rtInfo.find(old_api_map_key_order);
            if (it_order != rtInfo.end()) {
                const auto order = it_order->second.as<ov::OldApiMapOrder>().value;
                pre_input.preprocess().convert_layout(order);
                rtInfo.erase(it_order);
            }
        }

        auto& results = function->get_results();
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            ov::RTMap& rtInfo = result->get_rt_info();
            const auto it = rtInfo.find(old_api_map_key_order);
            if (it == rtInfo.end())
                continue;

            const auto order = it->second.as<ov::OldApiMapOrder>().value;
            auto& post_output = prepost.output(i);
            post_output.postprocess().convert_layout(order);

            // remove old api once we applied it
            rtInfo.erase(it);
        }

        if (need_validate_nodes_and_infer_types)
            function->validate_nodes_and_infer_types();

        // Set version to 10
        function->set_rt_info<int64_t>(10, "version");

        function = prepost.build();
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(function, exts, is_new_api));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath,
                                const std::string& binPath,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool is_new_api,
                                bool enable_mmap) {
    auto exts = ov::legacy_convert::convert_extension(ov_exts);

    // Fix unicode name
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = ov::util::string_to_wstring(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif

    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{model_path};

    if (!binPath.empty()) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::wstring& weights_path = ov::util::string_to_wstring(binPath.c_str());
#else
        const std::string& weights_path = binPath;
#endif
        params.emplace_back(weights_path);
    }
    params.emplace_back(enable_mmap);

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }

    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, is_new_api);
    }

    const auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    std::string FEs;
    for (const auto& fe_name : manager.get_available_front_ends())
        FEs += fe_name + " ";
    IE_THROW(NetworkNotRead) << "Unable to read the model: " << modelPath
                             << " Please check that model format: " << fileExt
                             << " is supported and the model is correct."
                             << " Available frontends: " << FEs;
}

CNNNetwork details::ReadNetwork(const std::string& model,
                                const Blob::CPtr& weights,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool is_new_api,
                                bool frontendMode) {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;
    auto exts = ov::legacy_convert::convert_extension(ov_exts);

    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream};
    if (weights) {
        char* data = weights->cbuffer().as<char*>();
        std::shared_ptr<ov::AlignedBuffer> weights_buffer =
            std::make_shared<ov::SharedBuffer<Blob::CPtr>>(data, weights->byteSize(), weights);
        params.emplace_back(weights_buffer);
    }

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }
    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, is_new_api, frontendMode);
    }

    IE_THROW(NetworkNotRead)
        << "Unable to read the model. Please check if the model format is supported and model is correct.";
}

}  // namespace InferenceEngine
