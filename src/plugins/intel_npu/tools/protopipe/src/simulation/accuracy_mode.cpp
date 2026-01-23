//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "accuracy_mode.hpp"

#include <fstream>
#include <future>

#include "simulation/computation_builder.hpp"
#include "simulation/executor.hpp"
#include "simulation/layers_data.hpp"
#include "simulation/layer_validator.hpp"
#include "simulation/failed_iter.hpp"
#include "scenario/inference.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

#include <opencv2/gapi/gproto.hpp>  // cv::GCompileArgs

Result reportValidationResult(const std::vector<FailedIter>& failed_iters, const size_t total_iters);
void updateCriterion(ITermCriterion::Ptr* criterion, cv::util::optional<uint64_t> required_num_iterations);
void dumpIterOutput(const cv::Mat& mat, const Dump& dump, const size_t iter);

static std::vector<std::string> compareOutputs(
    const std::vector<cv::Mat>& ref_mats,
    const std::vector<cv::Mat>& tgt_mats,
    const InferDesc& infer,
    const AccuracySimulation::Options& opts) {

    std::vector<std::string> failed_list;

    auto default_metric = opts.global_metric ? opts.global_metric : std::make_shared<Norm>(0.0);
    auto per_layer_metrics = unpackWithDefault(
        opts.metrics_map.at(infer.tag),
        extractLayerNames(infer.output_layers),
        default_metric
    );

    for (size_t i = 0; i < infer.output_layers.size(); ++i) {
        const auto& layer = infer.output_layers[i];
        LayerValidator validator{infer.tag, layer.name, per_layer_metrics.at(layer.name)};
        auto result = validator(ref_mats[i], tgt_mats[i]);
        if (!result) {
            failed_list.push_back(std::move(result.str()));
        }
    }

    return failed_list;
}

static Result performValidation(
    const std::vector<std::vector<cv::Mat>>& ref_outputs,
    const std::vector<std::vector<cv::Mat>>& tgt_outputs,
    const InferDesc& infer,
    const AccuracySimulation::Options& opts) {

    std::vector<FailedIter> failed_iters;

    // NB: Use tgt_outputs.size() as iteration count to match validation mode behavior.
    // Reference outputs are cycled via modulo if fewer iterations are available.
    for (size_t i = 0; i < tgt_outputs.size(); ++i) {
        auto failed_list = compareOutputs(ref_outputs[i % ref_outputs.size()], tgt_outputs[i], infer, opts);
        if (!failed_list.empty()) {
            failed_iters.push_back(FailedIter{i, std::move(failed_list)});
        }
    }

    return reportValidationResult(failed_iters, tgt_outputs.size());
}

namespace {

struct InputDataVisitor {
    InputDataVisitor(const InferDesc& _infer, const AccuracySimulation::Options& _opts)
            : infer(_infer), opts(_opts), providers(infer.input_layers.size()), metas(infer.input_layers.size()) {
    }

    void operator()(std::monostate);
    void operator()(const std::string&);
    void operator()(const LayerVariantAttr<std::string>&);

    InferDesc infer;
    const AccuracySimulation::Options& opts;
    // NB: Relevant when input reference data already exists and need to
    // generate exactly the same amount of output data.
    // Note that this value must be the same for all models within stream.
    cv::util::optional<uint64_t> model_required_iterations;
    std::vector<IDataProvider::Ptr> providers;
    std::vector<Meta> metas;
};

void InputDataVisitor::operator()(std::monostate) {
    // NB: No path provided - generate input random data using initializers.
    const auto input_names = extractLayerNames(infer.input_layers);
    const auto& initializers = opts.initializers_map.at(infer.tag);

    auto default_initialzer =
        opts.global_initializer ? opts.global_initializer : std::make_shared<UniformGenerator>(0.0, 255.0);
    auto layer_initializers = unpackWithDefault(initializers, input_names, default_initialzer);
    providers = createRandomProviders(infer.input_layers, std::move(layer_initializers));
};

void InputDataVisitor::operator()(const LayerVariantAttr<std::string>&) {
    THROW_ERROR("Accuracy mode does not support per-layer input data paths."
                " Provide either a single directory, a single file, or omit input data to use random generation!");
};

void InputDataVisitor::operator()(const std::string& path_str) {
    // NB: Single path provided - either single file or directory.
    const auto input_names = extractLayerNames(infer.input_layers);
    const auto& initializers = opts.initializers_map.at(infer.tag);

    std::filesystem::path path{path_str};
    if (std::filesystem::exists(path)) {
        // NB: Provided path exists - upload input data from there.
        LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag << " exists - data will be uploaded"
                   << std::endl;
        auto layers_data = uploadData(path, infer.tag, infer.input_layers, LayersType::INPUT);
        // NB: The Number of iterations for every layer is ALWAYS the same.
        model_required_iterations = cv::util::make_optional(layers_data.begin()->second.size());
        providers = createConstantProviders(std::move(layers_data), input_names);
    } else {
        // NB: Provided path doesn't exist - fall back to random data generation.
        LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag
                   << " doesn't exist - using random data" << std::endl;
        auto default_initialzer =
                opts.global_initializer ? opts.global_initializer : std::make_shared<UniformGenerator>(0.0, 255.0);
        auto layer_initializers = unpackWithDefault(initializers, input_names, default_initialzer);
        providers = createRandomProviders(infer.input_layers, std::move(layer_initializers));
    }
}

struct OutputDataVisitor {
    OutputDataVisitor(const InferDesc& _infer, const AccuracySimulation::Options& _opts)
            : infer(_infer), opts(_opts), metas(infer.output_layers.size()) {
    }

    void operator()(std::monostate);
    void operator()(std::string);
    void operator()(const LayerVariantAttr<std::string>&);

    InferDesc infer;
    const AccuracySimulation::Options& opts;
    std::vector<Meta> metas;
};

void OutputDataVisitor::operator()(std::monostate) {
    // NB: No path provided - outputs are stored in memory for validation only.
    for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
        metas[i].set(InferOutput{});
    }
}

void OutputDataVisitor::operator()(const LayerVariantAttr<std::string>&) {
    THROW_ERROR("Accuracy mode does not support per-layer output data paths."
                " Provide either a single directory, a single file, or omit output data to skip dumping!");
}

void OutputDataVisitor::operator()(std::string path_str) {
    // NB: Single path provided - creates _REFERENCE and _TARGET dump paths for dual-device comparison.

    // NB: Strip trailing path separators before appending suffixes.
    while (path_str.back() == '\\' || path_str.back() == '/') {
        path_str.pop_back();
    }
    std::filesystem::path ref_root{path_str + "_REFERENCE"};
    std::filesystem::path tgt_root{path_str + "_TARGET"};

    const auto layer_names = extractLayerNames(infer.output_layers);

    auto createDumpPaths = [&](const std::filesystem::path& root) -> std::vector<std::filesystem::path> {
        if (isDirectory(root)) {
            return createDirectoryLayout(root, layer_names);
        }
        if (infer.output_layers.size() > 1) {
            THROW_ERROR("Model: " << infer.tag
                                  << " must have exactly one output layer in order to dump output data to file: "
                                  << root);
        }
        return {root};
    };

    std::vector<std::filesystem::path> ref_dump_paths{createDumpPaths(ref_root)};
    std::vector<std::filesystem::path> tgt_dump_paths{createDumpPaths(tgt_root)};

    for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
        metas[i].set(DualDeviceDump{ref_dump_paths[i], tgt_dump_paths[i]});
    }
}

}  // anonymous namespace

class AccuracyStrategy : public IBuildStrategy {
public:
    explicit AccuracyStrategy(const AccuracySimulation::Options& opts);

    IBuildStrategy::InferBuildInfo build(const InferDesc& infer) override;

    // NB: If specified will force execution to perform exactly require_num_iterations
    // regardless what user specified.
    // Use case is when N input iterations are provided,
    // generate exactly the same amount of output iterations.
    // Another use case is when there is only single file provided
    // so only one input / output iteration must be generated.
    cv::optional<uint64_t> required_num_iterations;
    const AccuracySimulation::Options& opts;
    InferDesc current_infer;
};

AccuracyStrategy::AccuracyStrategy(const AccuracySimulation::Options& _opts): opts(_opts) {
}

IBuildStrategy::InferBuildInfo AccuracyStrategy::build(const InferDesc& infer) {
    current_infer = infer;
    const auto& input_data = opts.input_data_map.at(infer.tag);
    InputDataVisitor in_data_visitor{infer, opts};
    std::visit(in_data_visitor, input_data);
    // NB: Check if there is required number iterations for current model
    // and fail if it's different comparing to other models in stream.
    if (in_data_visitor.model_required_iterations) {
        const uint64_t required_iters_value = in_data_visitor.model_required_iterations.value();
        LOG_INFO() << "Model: " << infer.tag << " will perform at most " << required_iters_value << " iteration(s)"
                   << std::endl;
        if (!required_num_iterations) {
            required_num_iterations = in_data_visitor.model_required_iterations;
        } else {
            if (required_iters_value != required_num_iterations.value()) {
                THROW_ERROR("All models in stream are required to have the same number of iterations!");
            }
        }
    }

    const auto& output_data = opts.output_data_map.at(infer.tag);
    OutputDataVisitor out_data_visitor{infer, opts};
    std::visit(out_data_visitor, output_data);

    return {std::move(in_data_visitor.providers), std::move(in_data_visitor.metas), std::move(out_data_visitor.metas)};
}

namespace {

enum class DeviceType {
    Reference,
    Target
};

class SyncSimulation : public SyncCompiled {
public:
    SyncSimulation(cv::GCompiled&& ref_compiled, cv::GCompiled&& tgt_compiled,
                   std::vector<DummySource::Ptr>&& ref_sources, std::vector<DummySource::Ptr>&& tgt_sources,
                   std::vector<Meta>&& out_meta,
                   cv::util::optional<uint64_t> required_num_iterations,
                   const AccuracySimulation::Options& opts,
                   const InferDesc& infer);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GCompiled& pipeline, DeviceType device_type);

    SyncExecutor m_ref_exec;
    SyncExecutor m_tgt_exec;
    std::vector<DummySource::Ptr> m_ref_sources;
    std::vector<DummySource::Ptr> m_tgt_sources;
    std::vector<Meta> m_out_meta;
    cv::optional<uint64_t> m_required_num_iterations;
    const AccuracySimulation::Options m_opts;
    const InferDesc m_infer;

    std::vector<std::vector<cv::Mat>> m_ref_out_iter;
    std::vector<std::vector<cv::Mat>> m_tgt_out_iter;

    size_t m_ref_iter_idx;
    size_t m_tgt_iter_idx;
};

class PipelinedSimulation : public PipelinedCompiled {
public:
    PipelinedSimulation(cv::GStreamingCompiled&& ref_compiled, cv::GStreamingCompiled&& tgt_compiled,
                        std::vector<DummySource::Ptr>&& ref_sources, std::vector<DummySource::Ptr>&& tgt_sources,
                        std::vector<Meta>&& out_meta,
                        cv::util::optional<uint64_t> required_num_iterations,
                        const AccuracySimulation::Options& opts,
                        const InferDesc& infer);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GStreamingCompiled& pipeline, DeviceType device_type);

    PipelinedExecutor m_ref_exec;
    PipelinedExecutor m_tgt_exec;
    std::vector<DummySource::Ptr> m_ref_sources;
    std::vector<DummySource::Ptr> m_tgt_sources;
    std::vector<Meta> m_out_meta;
    cv::optional<uint64_t> m_required_num_iterations;
    const AccuracySimulation::Options m_opts;
    const InferDesc m_infer;

    std::vector<std::vector<cv::Mat>> m_ref_out_iter;
    std::vector<std::vector<cv::Mat>> m_tgt_out_iter;

    size_t m_ref_iter_idx;
    size_t m_tgt_iter_idx;
};

//////////////////////////////// SyncSimulation ////////////////////////////////
SyncSimulation::SyncSimulation(cv::GCompiled&& ref_compiled, cv::GCompiled&& tgt_compiled,
                               std::vector<DummySource::Ptr>&& ref_sources, std::vector<DummySource::Ptr>&& tgt_sources,
                               std::vector<Meta>&& out_meta,
                               cv::util::optional<uint64_t> required_num_iterations,
                               const AccuracySimulation::Options& opts,
                               const InferDesc& infer)
        : m_ref_exec(std::move(ref_compiled)),
          m_tgt_exec(std::move(tgt_compiled)),
          m_ref_sources(std::move(ref_sources)),
          m_tgt_sources(std::move(tgt_sources)),
          m_infer(std::move(infer)),
          m_out_meta(std::move(out_meta)),
          m_opts(std::move(opts)),
          m_ref_iter_idx(0u),
          m_tgt_iter_idx(0u),
          m_required_num_iterations(required_num_iterations) {
}

Result SyncSimulation::run(ITermCriterion::Ptr criterion) {
    updateCriterion(&criterion, m_required_num_iterations);
    auto tgt_criterion = criterion->clone();

    for (auto& src : m_ref_sources) src->reset();
    for (auto& src : m_tgt_sources) src->reset();

    // NB: Run REF and TGT pipelines in parallel, then validate outputs.
    auto ref_future = std::async(std::launch::async, [this, criterion]() {
        m_ref_exec.runLoop([this](cv::GCompiled& pipeline) {
            return this->process(pipeline, DeviceType::Reference);
        }, criterion);
    });

    auto tgt_future = std::async(std::launch::async, [this, tgt_criterion]() {
        m_tgt_exec.runLoop([this](cv::GCompiled& pipeline) {
            return this->process(pipeline, DeviceType::Target);
        }, tgt_criterion);
    });

    ref_future.get();
    tgt_future.get();

    auto validation_result = performValidation(
        m_ref_out_iter,
        m_tgt_out_iter,
        m_infer,
        m_opts
    );

    if (!validation_result) {
        return validation_result;
    }

    std::stringstream ss;
    ss << "Accuracy validation passed - Ref: " << m_ref_iter_idx
       << " iterations, Tgt: " << m_tgt_iter_idx << " iterations";
    return Success{ss.str()};
}

bool SyncSimulation::process(cv::GCompiled& pipeline, DeviceType device_type) {
    auto& sources  = (device_type == DeviceType::Reference) ? m_ref_sources  : m_tgt_sources;
    auto& iter_idx = (device_type == DeviceType::Reference) ? m_ref_iter_idx : m_tgt_iter_idx;
    auto& out_iter = (device_type == DeviceType::Reference) ? m_ref_out_iter : m_tgt_out_iter;

    std::vector<cv::Mat> out_mats(m_out_meta.size());

    auto pipeline_outputs = cv::gout();
    for (auto& out_mat : out_mats) {
        pipeline_outputs += cv::gout(out_mat);
    }

    cv::GRunArgs pipeline_inputs;
    pipeline_inputs.reserve(sources.size());
    for (auto src : sources) {
        cv::gapi::wip::Data data;
        src->pull(data);
        pipeline_inputs.push_back(std::move(data));
    }

    pipeline(std::move(pipeline_inputs), std::move(pipeline_outputs));

    for (size_t i = 0; i < out_mats.size(); ++i) {
        if (m_out_meta[i].has<DualDeviceDump>()) {
            const auto& dump = m_out_meta[i].get<DualDeviceDump>();
            auto dump_path = (device_type == DeviceType::Reference) ? dump.reference_path : dump.target_path;
            dumpIterOutput(out_mats[i], Dump{dump_path}, iter_idx);
        }
    }

    out_iter.push_back(std::move(out_mats));

    ++iter_idx;
    return true;
}

//////////////////////////////// PipelinedSimulation ////////////////////////////////
PipelinedSimulation::PipelinedSimulation(cv::GStreamingCompiled&& ref_compiled, cv::GStreamingCompiled&& tgt_compiled,
                                         std::vector<DummySource::Ptr>&& ref_sources, std::vector<DummySource::Ptr>&& tgt_sources,
                                         std::vector<Meta>&& out_meta,
                                         cv::util::optional<uint64_t> required_num_iterations,
                                         const AccuracySimulation::Options& opts,
                                         const InferDesc& infer)
        : m_ref_exec(std::move(ref_compiled)),
          m_tgt_exec(std::move(tgt_compiled)),
          m_ref_sources(std::move(ref_sources)),
          m_tgt_sources(std::move(tgt_sources)),
          m_out_meta(std::move(out_meta)),
          m_required_num_iterations(required_num_iterations),
          m_opts(std::move(opts)),
          m_infer(std::move(infer)),
          m_ref_iter_idx(0u),
          m_tgt_iter_idx(0u) {
}

Result PipelinedSimulation::run(ITermCriterion::Ptr criterion) {
    updateCriterion(&criterion, m_required_num_iterations);
    auto tgt_criterion = criterion->clone();

    // NB: Run REF and TGT streaming pipelines in parallel, then validate outputs.
    auto ref_future = std::async(std::launch::async, [this, criterion]() {
        auto ref_pipeline_inputs = cv::gin();
        for (auto source : m_ref_sources) {
            ref_pipeline_inputs += cv::gin(static_cast<cv::gapi::wip::IStreamSource::Ptr>(source));
        }
        m_ref_exec.runLoop(std::move(ref_pipeline_inputs), [this](cv::GStreamingCompiled& pipeline) {
            return this->process(pipeline, DeviceType::Reference);
        }, criterion);
    });

    auto tgt_future = std::async(std::launch::async, [this, tgt_criterion]() {
        auto tgt_pipeline_inputs = cv::gin();
        for (auto source : m_tgt_sources) {
            tgt_pipeline_inputs += cv::gin(static_cast<cv::gapi::wip::IStreamSource::Ptr>(source));
        }
        m_tgt_exec.runLoop(std::move(tgt_pipeline_inputs), [this](cv::GStreamingCompiled& pipeline) {
            return this->process(pipeline, DeviceType::Target);
        }, tgt_criterion);
    });

    ref_future.get();
    tgt_future.get();

    auto validation_result = performValidation(
        m_ref_out_iter,
        m_tgt_out_iter,
        m_infer,
        m_opts
    );

    if (!validation_result) {
        return validation_result;
    }

    std::stringstream ss;
    ss << "Accuracy validation passed - Ref: " << m_ref_iter_idx
       << " iterations, Tgt: " << m_tgt_iter_idx << " iterations";
    return Success{ss.str()};
}

bool PipelinedSimulation::process(cv::GStreamingCompiled& pipeline, DeviceType device_type) {
    auto& iter_idx = (device_type == DeviceType::Reference) ? m_ref_iter_idx : m_tgt_iter_idx;
    auto& out_iter = (device_type == DeviceType::Reference) ? m_ref_out_iter : m_tgt_out_iter;

    std::vector<cv::optional<cv::Mat>> opt_mats(m_out_meta.size());

    cv::GOptRunArgsP pipeline_outputs;
    for (auto& opt_mat : opt_mats) {
        pipeline_outputs.emplace_back(cv::gout(opt_mat)[0]);
    }

    const bool has_data = pipeline.pull(std::move(pipeline_outputs));

    if (has_data) {
        std::vector<cv::Mat> out_mats;
        out_mats.reserve(opt_mats.size());

        for (size_t i = 0; i < m_out_meta.size(); ++i) {
            ASSERT(opt_mats[i].has_value());

            if (m_out_meta[i].has<DualDeviceDump>()) {
                const auto& dump = m_out_meta[i].get<DualDeviceDump>();
                auto dump_path = (device_type == DeviceType::Reference) ? dump.reference_path : dump.target_path;
                dumpIterOutput(opt_mats[i].value(), Dump{dump_path}, iter_idx);
            }

            out_mats.push_back(std::move(opt_mats[i].value()));
        }

        out_iter.push_back(std::move(out_mats));
        ++iter_idx;
    }

    return has_data;
}

}  // anonymous namespace

static void changeDeviceParam(InferenceParamsMap& params, const std::string& device_name, const std::string& npu_compiler_type) {
    for (auto& [tag, inference_params] : params) {
        if (auto* ov = std::get_if<OpenVINOParams>(&inference_params)) {
            ov->device = device_name;

            if (device_name == "NPU") {
                if (ov->config.find("NPU_COMPILER_TYPE") == ov->config.end()) {
                    ov->config.emplace("NPU_COMPILER_TYPE", npu_compiler_type);
                }
            } else {
                ov->config.erase("NPU_COMPILER_TYPE");
            }
        } else if (auto* onnx = std::get_if<ONNXRTParams>(&inference_params)) {
            if (auto* ov_ep = std::get_if<ONNXRTParams::OpenVINO>(&onnx->ep)) {
                ov_ep->params_map["device_type"] = device_name;
            } else {
                onnx->ep = ONNXRTParams::OpenVINO{{{"device_type", device_name}}};
            }
        }
    }
}

AccuracySimulation::AccuracySimulation(Simulation::Config&& cfg, AccuracySimulation::Options&& opts)
        : Simulation(std::move(cfg)),
          m_opts(std::move(opts)),
          m_strategy(std::make_shared<AccuracyStrategy>(m_opts)),
          m_comp(ComputationBuilder{m_strategy}.build(m_cfg.graph, m_cfg.params, {false /* add performance meta */})) {
}

std::shared_ptr<PipelinedCompiled> AccuracySimulation::compilePipelined(DummySources&& ref_sources, DummySources&& tgt_sources,
                                                                        cv::GCompileArgs&& ref_compile_args, cv::GCompileArgs&& tgt_compile_args) {
    auto ref_descr = descr_of(ref_sources);
    auto tgt_descr = descr_of(tgt_sources);

    auto ref_future = std::async(std::launch::async, [this, ref_descr = std::move(ref_descr), ref_compile_args = std::move(ref_compile_args)]() mutable {
        return m_comp.compileStreaming(std::move(ref_descr), std::move(ref_compile_args));
    });

    auto tgt_future = std::async(std::launch::async, [this, tgt_descr = std::move(tgt_descr), tgt_compile_args = std::move(tgt_compile_args)]() mutable {
        return m_comp.compileStreaming(std::move(tgt_descr), std::move(tgt_compile_args));
    });

    auto ref_compiled = ref_future.get();
    auto tgt_compiled = tgt_future.get();

    auto out_meta = m_comp.getOutMeta();

    return std::make_shared<PipelinedSimulation>(std::move(ref_compiled), std::move(tgt_compiled),
                                                 std::move(ref_sources), std::move(tgt_sources),
                                                 std::move(out_meta), m_strategy->required_num_iterations,
                                                 std::move(m_opts), m_strategy->current_infer);
}

std::shared_ptr<PipelinedCompiled> AccuracySimulation::compilePipelined(const bool drop_frames) {
    changeDeviceParam(m_cfg.params, m_opts.tgt_device, m_opts.npu_compiler_type);
    auto tgt_compile_args = cv::compile_args(getNetworksPackage());
    changeDeviceParam(m_cfg.params, m_opts.ref_device, m_opts.npu_compiler_type);
    auto ref_compile_args = cv::compile_args(getNetworksPackage());

    // NB: Create separate sources for REF and TGT to avoid shared state issues.
    auto ref_sources = createSources(drop_frames);
    auto tgt_sources = createSources(drop_frames);

    return compilePipelined(std::move(ref_sources), std::move(tgt_sources),
                            std::move(ref_compile_args), std::move(tgt_compile_args));
}

std::shared_ptr<SyncCompiled> AccuracySimulation::compileSync(DummySources&& ref_sources, DummySources&& tgt_sources,
                                                            cv::GCompileArgs&& ref_compile_args, cv::GCompileArgs&& tgt_compile_args) {
    auto ref_descr = descr_of(ref_sources);
    auto tgt_descr = descr_of(tgt_sources);

    auto ref_future = std::async(std::launch::async, [this, ref_descr = std::move(ref_descr), ref_compile_args = std::move(ref_compile_args)]() mutable {
        return m_comp.compile(std::move(ref_descr), std::move(ref_compile_args));
    });

    auto tgt_future = std::async(std::launch::async, [this, tgt_descr = std::move(tgt_descr), tgt_compile_args = std::move(tgt_compile_args)]() mutable {
        return m_comp.compile(std::move(tgt_descr), std::move(tgt_compile_args));
    });

    auto ref_compiled = ref_future.get();
    auto tgt_compiled = tgt_future.get();

    auto out_meta = m_comp.getOutMeta();

    return std::make_shared<SyncSimulation>(std::move(ref_compiled), std::move(tgt_compiled),
                                            std::move(ref_sources), std::move(tgt_sources),
                                            std::move(out_meta), m_strategy->required_num_iterations,
                                            std::move(m_opts), m_strategy->current_infer);
}

std::shared_ptr<SyncCompiled> AccuracySimulation::compileSync(const bool drop_frames) {
    changeDeviceParam(m_cfg.params, m_opts.tgt_device, m_opts.npu_compiler_type);
    auto tgt_compile_args = cv::compile_args(getNetworksPackage());
    changeDeviceParam(m_cfg.params, m_opts.ref_device, m_opts.npu_compiler_type);
    auto ref_compile_args = cv::compile_args(getNetworksPackage());

    // NB: Create separate sources for REF and TGT to avoid shared state issues.
    auto ref_sources = createSources(drop_frames);
    auto tgt_sources = createSources(drop_frames);

    return compileSync(std::move(ref_sources), std::move(tgt_sources),
                       std::move(ref_compile_args), std::move(tgt_compile_args));
}
