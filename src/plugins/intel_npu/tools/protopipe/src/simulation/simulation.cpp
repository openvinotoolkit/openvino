//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/simulation.hpp"

#include "scenario/inference.hpp"
#include "utils/error.hpp"

#include <opencv2/gapi/infer/onnx.hpp>  // onnx::Params
#include <opencv2/gapi/infer/ov.hpp>    // ov::Params

static cv::gapi::GNetPackage getNetPackage(const std::string& tag, const OpenVINOParams& params) {
    using P = cv::gapi::ov::Params<cv::gapi::Generic>;
    std::unique_ptr<P> network;
    if (std::holds_alternative<OpenVINOParams::ModelPath>(params.path)) {
        const auto& model_path = std::get<OpenVINOParams::ModelPath>(params.path);
        network = std::make_unique<P>(tag, model_path.model, model_path.bin, params.device);
    } else {
        GAPI_Assert(std::holds_alternative<OpenVINOParams::BlobPath>(params.path));
        const auto& blob_path = std::get<OpenVINOParams::BlobPath>(params.path);
        network = std::make_unique<P>(tag, blob_path.blob, params.device);
    }

    network->cfgPluginConfig(params.config);
    network->cfgNumRequests(params.nireq);

    // NB: Pre/Post processing can be configured only for Model case.
    if (std::holds_alternative<OpenVINOParams::ModelPath>(params.path)) {
        if (std::holds_alternative<int>(params.output_precision)) {
            network->cfgOutputTensorPrecision(std::get<int>(params.output_precision));
        } else if (std::holds_alternative<AttrMap<int>>(params.output_precision)) {
            network->cfgOutputTensorPrecision(std::get<AttrMap<int>>(params.output_precision));
        }

        if (std::holds_alternative<std::string>(params.input_layout)) {
            network->cfgInputTensorLayout(std::get<std::string>(params.input_layout));
        } else if (std::holds_alternative<AttrMap<std::string>>(params.input_layout)) {
            network->cfgInputTensorLayout(std::get<AttrMap<std::string>>(params.input_layout));
        }

        if (std::holds_alternative<std::string>(params.output_layout)) {
            network->cfgOutputTensorLayout(std::get<std::string>(params.output_layout));
        } else if (std::holds_alternative<AttrMap<std::string>>(params.output_layout)) {
            network->cfgOutputTensorLayout(std::get<AttrMap<std::string>>(params.output_layout));
        }

        if (std::holds_alternative<std::string>(params.input_model_layout)) {
            network->cfgInputModelLayout(std::get<std::string>(params.input_model_layout));
        } else if (std::holds_alternative<AttrMap<std::string>>(params.input_model_layout)) {
            network->cfgInputModelLayout(std::get<AttrMap<std::string>>(params.input_model_layout));
        }

        if (std::holds_alternative<std::string>(params.output_model_layout)) {
            network->cfgOutputModelLayout(std::get<std::string>(params.output_model_layout));
        } else if (std::holds_alternative<AttrMap<std::string>>(params.output_model_layout)) {
            network->cfgOutputModelLayout(std::get<AttrMap<std::string>>(params.output_model_layout));
        }
    }
    return cv::gapi::networks(*network);
}

static void cfgExecutionProvider(cv::gapi::onnx::Params<cv::gapi::Generic>& network,
                                 const ONNXRTParams::OpenVINO& ovep) {
    network.cfgAddExecutionProvider(cv::gapi::onnx::ep::OpenVINO{ovep.params_map});
}

static void cfgExecutionProvider(cv::gapi::onnx::Params<cv::gapi::Generic>& network, const ONNXRTParams::EP& ep) {
    // NB: Nothing to configure for default MLAS EP
    if (std::holds_alternative<std::monostate>(ep)) {
        return;
    }
    // TODO: Extend for any other available execution provider
    ASSERT(std::holds_alternative<ONNXRTParams::OpenVINO>(ep));
    cfgExecutionProvider(network, std::get<ONNXRTParams::OpenVINO>(ep));
}

static cv::gapi::GNetPackage getNetPackage(const std::string& tag, const ONNXRTParams& params) {
    cv::gapi::onnx::Params<cv::gapi::Generic> network{tag, params.model_path};
    network.cfgSessionOptions(params.session_options);
    if (params.opt_level.has_value()) {
        network.cfgOptLevel(params.opt_level.value());
    }
    cfgExecutionProvider(network, params.ep);
    return cv::gapi::networks(network);
}

static cv::gapi::GNetPackage getNetPackage(const std::string& tag, const InferenceParams& params) {
    if (std::holds_alternative<OpenVINOParams>(params)) {
        return getNetPackage(tag, std::get<OpenVINOParams>(params));
    }
    ASSERT(std::holds_alternative<ONNXRTParams>(params));
    return getNetPackage(tag, std::get<ONNXRTParams>(params));
}

cv::gapi::GNetPackage Simulation::getNetworksPackage() const {
    cv::gapi::GNetPackage networks;
    for (const auto& [tag, params] : m_cfg.params) {
        networks += getNetPackage(tag, params);
    }
    return networks;
}

Simulation::Simulation(Config&& cfg): m_cfg(std::move(cfg)){};

std::vector<DummySource::Ptr> Simulation::createSources(const bool drop_frames) {
    auto src = std::make_shared<DummySource>(m_cfg.frames_interval_in_us, drop_frames,
                                             m_cfg.disable_high_resolution_timer);
    return {src};
};

std::shared_ptr<PipelinedCompiled> Simulation::compilePipelined(const bool drop_frames) {
    if (drop_frames) {
        THROW_ERROR("Pipelined simulation doesn't support frames drop!");
    }
    // NB: Hardcoded for pipelining mode as the best option
    auto compile_args = cv::compile_args(getNetworksPackage());
    compile_args += cv::compile_args(cv::gapi::streaming::queue_capacity{1u});
    return compilePipelined(createSources(drop_frames), std::move(compile_args));
}

std::shared_ptr<SyncCompiled> Simulation::compileSync(const bool drop_frames) {
    auto compile_args = cv::compile_args(getNetworksPackage());
    return compileSync(createSources(drop_frames), std::move(compile_args));
}

std::shared_ptr<PipelinedCompiled> Simulation::compilePipelined(DummySources&&, cv::GCompileArgs&&) {
    THROW_ERROR("Not implemented!");
};

std::shared_ptr<SyncCompiled> Simulation::compileSync(DummySources&&, cv::GCompileArgs&&) {
    THROW_ERROR("Not implemented!");
}
