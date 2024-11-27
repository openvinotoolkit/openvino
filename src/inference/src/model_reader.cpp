// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_reader.hpp"

#include "itt.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "transformations/utils/utils.hpp"

namespace {
ov::element::Type to_legacy_type(const ov::element::Type& legacy_type, bool input) {
    if (input) {
        return legacy_type == ov::element::f16 ? ov::element::f32 : legacy_type;
    } else {
        if (legacy_type == ov::element::i64 || legacy_type == ov::element::u64 || legacy_type == ov::element::i32 ||
            legacy_type == ov::element::u32) {
            return ov::element::i32;
        } else if (legacy_type != ov::element::f32) {
            return ov::element::f32;
        }
    }

    return legacy_type;
}

void update_v10_model(std::shared_ptr<ov::Model>& model, bool frontendMode = false) {
    // only for IR cases we need preprocessing or postprocessing steps
    if (model->has_rt_info("version") && model->get_rt_info<int64_t>("version") == 10) {
        IR_READER_SCOPE(ir10_new_api);

        ov::preprocess::PrePostProcessor prepost(model);
        std::unordered_map<std::string, std::shared_ptr<ov::descriptor::Tensor>> leaf_names;
        const auto inputs = model->inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!frontendMode) {
                const auto ov_type = inputs[i].get_element_type();
                const auto legacy_type = to_legacy_type(ov_type, true);
                prepost.input(i).tensor().set_element_type(legacy_type);
            }
            for (const auto& name : inputs[i].get_names()) {
                OPENVINO_ASSERT(leaf_names.find(name) == leaf_names.end(),
                                "Model tensor names have collisions.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.emplace(name, inputs[i].get_tensor_ptr());
            }
        }

        const auto outputs = model->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!frontendMode) {
                const auto ov_type = outputs[i].get_element_type();
                const auto legacy_type = to_legacy_type(ov_type, false);
                prepost.output(i).tensor().set_element_type(legacy_type);
            }
            for (const auto& name : outputs[i].get_names()) {
                auto tensor_it = leaf_names.find(name);
                OPENVINO_ASSERT(tensor_it == leaf_names.end() || tensor_it->second == outputs[i].get_tensor_ptr(),
                                "Model tensor names have collisions.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.emplace(name, outputs[i].get_tensor_ptr());
            }
        }

        // in order to support the following scenarios for IR v10 cases:
        // ov::Model f = ie.read_model(..);
        // f.input("input_operation_name");
        // f.output("output_operation_name");
        // f.add_output("operation_name[].port_index]");
        // f.reshape({ { "input_operation_name", ov::PartialShape{} } });
        // we need to add operation names as tensor names for inputs and outputs
        {
            for (const auto& result : model->get_results()) {
                OPENVINO_SUPPRESS_DEPRECATED_START
                // Note, upon removal of 'create_ie_output_name', just move it to this file as a local function
                // we still need to add operation names as tensor names for outputs for IR v10
                auto res_name = ov::op::util::create_ie_output_name(result->input_value(0));
                OPENVINO_SUPPRESS_DEPRECATED_END
                OPENVINO_ASSERT(leaf_names.find(res_name) == leaf_names.end() ||
                                    result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.emplace(res_name, nullptr);
                result->output(0).get_tensor().add_names({std::move(res_name)});
            }
            for (const auto& param : model->get_parameters()) {
                const auto& param_name = param->get_friendly_name();
                OPENVINO_ASSERT(leaf_names.find(param_name) == leaf_names.end() ||
                                    param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                                "Model operation names have collisions with tensor names.",
                                " Please use MO to generate new IR version, it should allow to avoid the issue");
                leaf_names.emplace(param_name, nullptr);
                param->output(0).get_tensor().add_names({param_name});
            }
        }

        model = prepost.build();
    }
}
}  // namespace

namespace ov {
namespace util {

std::shared_ptr<ov::Model> read_model(const std::string& modelPath,
                                      const std::string& binPath,
                                      const std::vector<ov::Extension::Ptr>& extensions,
                                      bool enable_mmap) {
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
        FE->add_extension(extensions);
        inputModel = FE->load(params);
    }

    if (inputModel) {
        auto model = FE->convert(inputModel);
        update_v10_model(model);
        return model;
    }

    const auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    std::string FEs;
    for (const auto& fe_name : manager.get_available_front_ends())
        FEs += fe_name + " ";
    OPENVINO_THROW("Unable to read the model: ",
                   modelPath,
                   " Please check that model format: ",
                   fileExt,
                   " is supported and the model is correct.",
                   " Available frontends: ",
                   FEs);
}

std::shared_ptr<ov::Model> read_model(const std::string& model,
                                      const ov::Tensor& weights,
                                      const std::vector<ov::Extension::Ptr>& ov_exts,
                                      bool frontendMode) {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;

    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream};
    if (weights) {
        std::shared_ptr<ov::AlignedBuffer> weights_buffer =
            std::make_shared<ov::SharedBuffer<ov::Tensor>>(reinterpret_cast<char*>(weights.data()),
                                                           weights.get_byte_size(),
                                                           weights);
        params.emplace_back(weights_buffer);
    }

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }
    if (inputModel) {
        auto model = FE->convert(inputModel);
        update_v10_model(model);
        return model;
    }

    OPENVINO_THROW("Unable to read the model. Please check if the model format is supported and model is correct.");
}

std::shared_ptr<ov::Model> read_model(const std::shared_ptr<AlignedBuffer>& model,
                                      const std::shared_ptr<AlignedBuffer>& weights,
                                      const std::vector<ov::Extension::Ptr>& ov_exts) {
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{model};
    if (weights) {
        params.emplace_back(weights);
    }

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }
    if (inputModel) {
        auto model = FE->convert(inputModel);
        update_v10_model(model);
        return model;
    }

    OPENVINO_THROW(
        "[ CORE ] Unable to read the model. Please check if the model format is supported and model is correct.");
}

}  // namespace util
}  // namespace ov
