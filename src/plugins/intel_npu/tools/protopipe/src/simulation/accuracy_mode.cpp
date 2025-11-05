// TODO: Copyright tag

#include "accuracy_mode.hpp"

#include <fstream>
#include <future>

#include "simulation/computation_builder.hpp"
#include "simulation/executor.hpp"
#include "simulation/layers_data.hpp"
#include "simulation/layer_validator.hpp"
#include "scenario/inference.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

#include <opencv2/gapi/gproto.hpp>  // cv::GCompileArgs

struct FailedIter {
    size_t iter_idx;
    std::vector<std::string> reasons;
};

static std::vector<std::string> compareOutputs(
    const std::vector<cv::Mat>& ref_mats,
    const std::vector<cv::Mat>& tgt_mats,
    const std::vector<Meta>& out_meta,
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

static Result reportValidationResult(const std::vector<FailedIter>& failed_iters, const size_t total_iters) {
    std::stringstream ss;
    if (!failed_iters.empty()) {
        const auto kItersToShow = 10u;
        const auto kLimit = failed_iters.size() < kItersToShow ? failed_iters.size() : kItersToShow;
        ss << "Accuraccy check failed on " << failed_iters.size() << " iteration(s)"
           << " (first " << kLimit << "):";
        ss << "\n";
        for (uint32_t i = 0; i < kLimit; ++i) {
            ss << "Iteration " << failed_iters[i].iter_idx << ":\n";
            for (const auto& reason : failed_iters[i].reasons) {
                ss << "  " << reason << "\n";
            }
        }
        return Error{ss.str()};
    }
    ss << "Validation has passed for " << total_iters << " iteration(s)";
    return Success{ss.str()};
}

static Result performValidation(
    const std::vector<std::vector<cv::Mat>>& ref_outputs,
    const std::vector<std::vector<cv::Mat>>& tgt_outputs,
    const std::vector<Meta>& out_meta,
    const InferDesc& infer,
    const AccuracySimulation::Options& opts) {

    std::cout << "Performing validation\n";

    std::vector<FailedIter> failed_iters;
    size_t num_iters = std::min(ref_outputs[0].size(), tgt_outputs[0].size());

    for (size_t iter = 0; iter < num_iters; ++iter) {
        std::vector<cv::Mat> ref_iter_mats;
        std::vector<cv::Mat> tgt_iter_mats;

        for (size_t layer = 0; layer < ref_outputs.size(); ++layer) {
            ref_iter_mats.push_back(ref_outputs[layer][iter]);
            tgt_iter_mats.push_back(tgt_outputs[layer][iter]);
        }

        auto failed_list = compareOutputs(ref_iter_mats, tgt_iter_mats, out_meta, infer, opts);
        if (!failed_list.empty()) {
            failed_iters.push_back(FailedIter{iter, std::move(failed_list)});
        }
    }

    return reportValidationResult(failed_iters, num_iters);
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
    // TODO: if the path is not passed for the input data, the input data will be kept in memory
};

void InputDataVisitor::operator()(const LayerVariantAttr<std::string>&) {
    THROW_ERROR("Reference mode requires output data path to be provided"
                " in form of either directory or single file!");
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
        // NB: Provided path doesn't exist - generate data and dump.
        LOG_INFO() << "Input data path: " << path << " for model: " << infer.tag
                   << " doesn't exist - input data will be generated and dumped" << std::endl;
        std::vector<std::filesystem::path> dump_path_vec;
        if (isDirectory(path)) {
            // NB: When the directory is provided, the number of input iterations to be generated aren't
            // bounded so the "random" providers will generate input data on every iteration that will
            // be dumped on the disk afterwards.
            dump_path_vec = createDirectoryLayout(path, input_names);
        } else {
            // NB: When the single file is provided, the execution must be limited to perform
            // only 1 iteration.
            model_required_iterations = cv::util::optional<uint64_t>(1ul);
            if (infer.input_layers.size() > 1) {
                THROW_ERROR("Model: " << infer.tag
                                      << " must have exactly one input layer in order to dump input data to file: "
                                      << path);
            }
            // NB: In case directories in that path don't exist.
            std::filesystem::create_directories(path.parent_path());
            dump_path_vec = {path};
        }
        auto default_initialzer =
                opts.global_initializer ? opts.global_initializer : std::make_shared<UniformGenerator>(0.0, 255.0);
        auto layer_initializers = unpackWithDefault(initializers, input_names, default_initialzer);
        providers = createRandomProviders(infer.input_layers, std::move(layer_initializers));
        for (uint32_t i = 0; i < infer.input_layers.size(); ++i) {
            metas[i].set(Dump{dump_path_vec[i]});
        }
    }
}

struct OutputDataVisitor {
    OutputDataVisitor(const InferDesc& _infer, const AccuracySimulation::Options& _opts)
            : infer(_infer), opts(_opts), metas(infer.output_layers.size()) {
    }

    void operator()(std::monostate);
    void operator()(const std::string&);
    void operator()(const LayerVariantAttr<std::string>&);

    InferDesc infer;
    const AccuracySimulation::Options& opts;
    std::vector<Meta> metas;
};

void OutputDataVisitor::operator()(std::monostate) {
    // TODO: It shouldn't fail if no output path is provided, it should just do nothing
}

void OutputDataVisitor::operator()(const LayerVariantAttr<std::string>&) {
    THROW_ERROR("Reference mode requires output data path to be provided"
                " in form of either directory or single file!");
}

// TODO: modify so it will dump the reference data from reference and target device
void OutputDataVisitor::operator()(const std::string& path_str) {
    // std::filesystem::path path{path_str};
    // // NB: It doesn't matter if path exist or not - regenerate and dump outputs anyway.
    // std::vector<std::filesystem::path> dump_path_vec;
    // if (isDirectory(path)) {
    //     dump_path_vec = createDirectoryLayout(path, extractLayerNames(infer.output_layers));
    // } else {
    //     if (infer.output_layers.size() > 1) {
    //         THROW_ERROR("Model: " << infer.tag
    //                               << " must have exactly one output layer in order to dump output data to file: "
    //                               << path);
    //     }
    //     dump_path_vec = {path};
    // }
    // for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
    //     const auto& layer = infer.output_layers[i];
    //     metas[i].set(Dump{dump_path_vec[i]});
    // }
    // auto default_metric = opts.global_metric ? opts.global_metric : std::make_shared<Norm>(0.0);
    // auto per_layer_metrics =
    //         unpackWithDefault(opts.metrics_map.at(infer.tag), extractLayerNames(infer.output_layers), default_metric);
    // std::filesystem::path path{path_str};
    // LOG_INFO() << "Reference output data path: " << path << " for model: " << infer.tag
    //            << " exists - data will be uploaded" << std::endl;
    // // TODO: change the actual uploadData stuff with the ref/tgt vectors
    // auto layers_data = uploadData(path, infer.tag, infer.output_layers, LayersType::OUTPUT);
    // for (uint32_t i = 0; i < infer.output_layers.size(); ++i) {
    //     const auto& layer = infer.output_layers[i];
    //     LayerValidator validator{infer.tag, layer.name, per_layer_metrics.at(layer.name)};
    //     metas[i].set(Validate{std::move(validator), layers_data.at(layer.name)});
    // }
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

static void updateCriterion(ITermCriterion::Ptr* criterion, cv::util::optional<uint64_t> required_num_iterations) {
    if (required_num_iterations.has_value()) {
        if (*criterion) {
            // NB: Limit user's termination criterion to perfom at most m_required_num_iterations
            *criterion = std::make_shared<CombinedCriterion>(
                    *criterion, std::make_shared<Iterations>(required_num_iterations.value()));
        } else {
            *criterion = std::make_shared<Iterations>(required_num_iterations.value());
        }
    }
}

static void dumpIterOutput(const cv::Mat& mat, const Dump& dump, const size_t iter) {
    auto dump_path = dump.path;
    if (isDirectory(dump.path)) {
        std::stringstream ss;
        ss << "iter_" << iter << ".bin";
        dump_path = dump_path / ss.str();
    }
    utils::writeToBinFile(dump_path.string(), mat);
};

namespace {

enum class DeviceType {
    Reference,
    Target
};

class SyncSimulation : public SyncCompiled {
public:
    SyncSimulation(cv::GCompiled&& ref_compiled, cv::GCompiled&& tgt_compiled, 
                   std::vector<DummySource::Ptr>&& sources, 
                   std::vector<Meta>&& ref_out_meta, std::vector<Meta>&& tgt_out_meta, 
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
    std::vector<Meta> m_ref_out_meta;
    std::vector<Meta> m_tgt_out_meta;
    cv::optional<uint64_t> m_required_num_iterations;
    const AccuracySimulation::Options m_opts;
    const InferDesc m_infer;

    std::vector<cv::Mat> m_ref_out_mats;
    std::vector<cv::Mat> m_tgt_out_mats;

    std::vector<std::vector<cv::Mat>> m_ref_outputs;
    std::vector<std::vector<cv::Mat>> m_tgt_outputs;
    size_t m_ref_iter_idx;
    size_t m_tgt_iter_idx;
};

class PipelinedSimulation : public PipelinedCompiled {
public:
    PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                        std::vector<Meta>&& out_meta, cv::util::optional<uint64_t> required_num_iterations);

    Result run(ITermCriterion::Ptr criterion) override;

private:
    bool process(cv::GStreamingCompiled& pipeline);

    PipelinedExecutor m_exec;
    std::vector<DummySource::Ptr> m_sources;
    std::vector<Meta> m_out_meta;
    std::vector<cv::optional<cv::Mat>> m_opt_mats;
    size_t m_iter_idx;
    cv::optional<uint64_t> m_required_num_iterations;
};

//////////////////////////////// SyncSimulation ////////////////////////////////
SyncSimulation::SyncSimulation(cv::GCompiled&& ref_compiled, cv::GCompiled&& tgt_compiled, std::vector<DummySource::Ptr>&& sources, 
                               std::vector<Meta>&& ref_out_meta, std::vector<Meta>&& tgt_out_meta, 
                               cv::util::optional<uint64_t> required_num_iterations, const AccuracySimulation::Options& opts,
                               const InferDesc& infer)
        : m_ref_exec(std::move(ref_compiled)),
          m_tgt_exec(std::move(tgt_compiled)),
          m_ref_sources(sources),
          m_tgt_sources(std::move(sources)),
          m_ref_out_meta(std::move(ref_out_meta)),
          m_tgt_out_meta(std::move(tgt_out_meta)),
          m_ref_out_mats(m_ref_out_meta.size()),
          m_tgt_out_mats(m_tgt_out_meta.size()),
          m_opts(std::move(opts)),
          m_infer(std::move(infer)),
          m_ref_iter_idx(0u),
          m_tgt_iter_idx(0u),
          m_required_num_iterations(required_num_iterations) {

    m_ref_outputs.resize(m_ref_out_meta.size());
    m_tgt_outputs.resize(m_tgt_out_meta.size());
}

Result SyncSimulation::run(ITermCriterion::Ptr criterion) {
    auto ref_criterion = criterion;
    auto tgt_criterion = criterion;

    updateCriterion(&ref_criterion, m_required_num_iterations);
    updateCriterion(&tgt_criterion, m_required_num_iterations);

    for (auto src : m_ref_sources) {
        src->reset();
    }
    for (auto src : m_tgt_sources) {
        src->reset();
    }

    auto ref_process = [this](cv::GCompiled& pipeline) -> bool {
        return this->process(pipeline, DeviceType::Reference);
    };

    auto tgt_process = [this](cv::GCompiled& pipeline) -> bool {
        return this->process(pipeline, DeviceType::Target);
    };

    auto ref_future = std::async(std::launch::async, [this, ref_process, ref_criterion]() {
        m_ref_exec.runLoop(ref_process, ref_criterion);
    });

    auto tgt_future = std::async(std::launch::async, [this, tgt_process, tgt_criterion]() {
        m_tgt_exec.runLoop(tgt_process, tgt_criterion);
    });

    ref_future.get();
    tgt_future.get();

    auto validation_result = performValidation(
        m_ref_outputs,
        m_tgt_outputs,
        m_ref_out_meta,
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
    auto& out_mats = (device_type == DeviceType::Reference) ? m_ref_out_mats : m_tgt_out_mats;
    auto& out_meta = (device_type == DeviceType::Reference) ? m_ref_out_meta : m_tgt_out_meta;
    auto& outputs  = (device_type == DeviceType::Reference) ? m_ref_outputs  : m_tgt_outputs;
    auto& iter_idx = (device_type == DeviceType::Reference) ? m_ref_iter_idx : m_tgt_iter_idx;

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
        outputs[i].push_back(out_mats[i].clone());
    }

    for (size_t i = 0; i < out_mats.size(); ++i) {
        if (out_meta[i].has<Dump>()) {
            const auto& dump = out_meta[i].get<Dump>();
            dumpIterOutput(out_mats[i], dump, iter_idx);
        }
    }

    ++iter_idx;
    return true;
}

//////////////////////////////// PipelinedSimulation ///////////////////////////////
PipelinedSimulation::PipelinedSimulation(cv::GStreamingCompiled&& compiled, std::vector<DummySource::Ptr>&& sources,
                                         std::vector<Meta>&& out_meta,
                                         cv::util::optional<uint64_t> required_num_iterations)
        : m_exec(std::move(compiled)),
          m_sources(std::move(sources)),
          m_out_meta(std::move(out_meta)),
          m_opt_mats(m_out_meta.size()),
          m_iter_idx(0u),
          m_required_num_iterations(required_num_iterations) {
}

Result PipelinedSimulation::run(ITermCriterion::Ptr criterion) {
    auto pipeline_inputs = cv::gin();
    for (auto source : m_sources) {
        pipeline_inputs += cv::gin(static_cast<cv::gapi::wip::IStreamSource::Ptr>(source));
    }
    using namespace std::placeholders;
    auto cb = std::bind(&PipelinedSimulation::process, this, _1);
    updateCriterion(&criterion, m_required_num_iterations);
    m_exec.runLoop(std::move(pipeline_inputs), cb, criterion);
    std::stringstream ss;
    ss << "Reference data has been generated for " << m_iter_idx << " iteration(s)";
    return Success{ss.str()};
};

bool PipelinedSimulation::process(cv::GStreamingCompiled& pipeline) {
    cv::GOptRunArgsP pipeline_outputs;
    for (auto& opt_mat : m_opt_mats) {
        pipeline_outputs.emplace_back(cv::gout(opt_mat)[0]);
    }
    const bool has_data = pipeline.pull(std::move(pipeline_outputs));
    for (size_t i = 0; i < m_out_meta.size(); ++i) {
        if (m_out_meta[i].has<Dump>()) {
            const auto& dump = m_out_meta[i].get<Dump>();
            ASSERT(m_opt_mats[i].has_value());
            dumpIterOutput(m_opt_mats[i].value(), dump, m_iter_idx);
        }
    }
    ++m_iter_idx;
    return has_data;
}

}  // anonymous namespace

static void changeDeviceParam(InferenceParamsMap& params, const std::string& device_name) {
    for (auto& [tag, inference_params] : params) {
        if (std::holds_alternative<OpenVINOParams>(inference_params)) {
            std::get<OpenVINOParams>(inference_params).device = device_name;
        }
    }
}

AccuracySimulation::AccuracySimulation(Simulation::Config&& cfg, AccuracySimulation::Options&& opts)
        : Simulation(std::move(cfg)),
          m_opts(std::move(opts)),
          m_strategy(std::make_shared<AccuracyStrategy>(m_opts)),
          m_comp(ComputationBuilder{m_strategy}.build(m_cfg.graph, m_cfg.params, {false /* add performance meta */})) {
}

std::shared_ptr<PipelinedCompiled> AccuracySimulation::compilePipelined(DummySources&& sources,
                                                                       cv::GCompileArgs&& compile_args) {
    auto compiled = m_comp.compileStreaming(descr_of(sources), std::move(compile_args));
    auto out_meta = m_comp.getOutMeta();
    return std::make_shared<PipelinedSimulation>(std::move(compiled), std::move(sources), std::move(out_meta),
                                                 m_strategy->required_num_iterations);
}

std::shared_ptr<SyncCompiled> AccuracySimulation::compileSync(DummySources&& sources, cv::GCompileArgs&& ref_compile_args, cv::GCompileArgs&& tgt_compile_args) {
    auto ref_compiled = m_comp.compile(descr_of(sources), std::move(ref_compile_args));
    auto ref_out_meta = m_comp.getOutMeta();

    for (auto src : sources) {
        src->reset();
    }

    auto tgt_compiled = m_comp.compile(descr_of(sources), std::move(tgt_compile_args));
    auto tgt_out_meta = m_comp.getOutMeta();

    return std::make_shared<SyncSimulation>(std::move(ref_compiled), std::move(tgt_compiled), std::move(sources), 
                                            std::move(ref_out_meta), std::move(tgt_out_meta), m_strategy->required_num_iterations, std::move(m_opts), m_strategy->current_infer);
}

std::shared_ptr<SyncCompiled> AccuracySimulation::compileSync(const bool drop_frames) {
    changeDeviceParam(m_cfg.params, m_opts.tgt_device);
    auto tgt_compile_args = cv::compile_args(getNetworksPackage());
    changeDeviceParam(m_cfg.params, m_opts.ref_device);
    auto ref_compile_args = cv::compile_args(getNetworksPackage());
    return compileSync(createSources(drop_frames), std::move(ref_compile_args), std::move(tgt_compile_args));
}
