// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Measures the cost of running a chain of infer requests of the same compiled model, the way NPUW runs
// repeated decoder blocks with NPUW_UNFOLD_IREQS=YES, against a single model with all blocks merged.
// See src/plugins/intel_npu/docs/execution-list-proposal.md.

#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/iremote_context.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "intel_npu/common/execution_list.hpp"

static constexpr char help_message[] = "Optional. Print the usage message.";

static constexpr char device_message[] = "Optional. Target device. Default value: NPU.";

static constexpr char mode_message[] =
    "Optional. Comma-separated list of modes to run. Default value: all.\n"
    "                                             unfold-async   - one request per block, start all, then wait all\n"
    "                                                              (NPUW_UNFOLD_IREQS + NPUW_FUNCALL_ASYNC)\n"
    "                                             unfold-sync    - one request per block, start and wait each\n"
    "                                             execution-list - one request per block, submitted as one\n"
    "                                                              execution list (NPU only)\n"
    "                                             reference      - one model with all blocks merged\n"
    "                                             all            - all of the above, execution-list only on NPU";

static constexpr char blocks_message[] = "Optional. Number of chained blocks. Default value: 32.";

static constexpr char hidden_message[] = "Optional. Hidden size of a block. Default value: 2560 (Qwen3-4B).";

static constexpr char ffn_message[] = "Optional. Intermediate size of a block. Default value: 9728 (Qwen3-4B).";

static constexpr char seq_message[] = "Optional. Number of tokens per inference. Default value: 1 (generate stage).";

static constexpr char ref_weights_message[] =
    "Optional. How the reference model receives the weights. Default value: param.\n"
    "                                             param - as inputs, the same way the block requests receive them\n"
    "                                             const - as constants, the compiler sees the weights";

static constexpr char niter_message[] = "Optional. Number of measured iterations per mode. Default value: 100.";

static constexpr char warmup_message[] = "Optional. Number of warm-up iterations per mode. Default value: 10.";

static constexpr char config_message[] =
    "Optional. Comma-separated KEY=VALUE properties passed to every compile_model call.\n"
    "                                             Example: -config \"NPU_TURBO=YES,PERF_COUNT=YES\"";

static constexpr char check_message[] =
    "Optional. After the last iteration of every mode, compare its output against a reference output computed once\n"
    "                                             before any mode runs. Default value: true.";

static constexpr char ref_device_message[] =
    "Optional. Device that computes the reference output for -check, using the merged model.\n"
    "                                             An empty value uses the device under test. Default value: CPU.";

static constexpr char tolerance_message[] =
    "Optional. Maximum allowed max-abs difference relative to the reference max-abs value. Default value: 0.05.";

static constexpr char rebind_message[] =
    "Optional. Before every iteration, move the block requests to another set of output tensors, the way NPUW\n"
    "                                             rebinds the KV cache between tokens. Exercises updating the tensors of\n"
    "                                             a recorded execution list. Default value: false.";

static constexpr char seed_message[] = "Optional. Seed for the generated weights and input. Default value: 42.";

DEFINE_bool(h, false, help_message);
DEFINE_string(d, "NPU", device_message);
DEFINE_string(mode, "all", mode_message);
DEFINE_uint32(blocks, 32, blocks_message);
DEFINE_uint32(hidden, 2560, hidden_message);
DEFINE_uint32(ffn, 9728, ffn_message);
DEFINE_uint32(seq, 1, seq_message);
DEFINE_string(ref_weights, "param", ref_weights_message);
DEFINE_uint32(niter, 100, niter_message);
DEFINE_uint32(warmup, 10, warmup_message);
DEFINE_string(config, "", config_message);
DEFINE_bool(check, true, check_message);
DEFINE_string(ref_device, "CPU", ref_device_message);
DEFINE_double(tolerance, 0.05, tolerance_message);
DEFINE_bool(rebind, false, rebind_message);
DEFINE_uint32(seed, 42, seed_message);

static void showUsage() {
    std::cout << "execution-list-bench [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:                             " << std::endl;
    std::cout << "    -h                                       " << help_message << std::endl;
    std::cout << "    -d                           <value>     " << device_message << std::endl;
    std::cout << "    -mode                        <value>     " << mode_message << std::endl;
    std::cout << "    -niter                       <value>     " << niter_message << std::endl;
    std::cout << "    -warmup                      <value>     " << warmup_message << std::endl;
    std::cout << "    -config                      <value>     " << config_message << std::endl;
    std::cout << "    -check                       <value>     " << check_message << std::endl;
    std::cout << "    -ref_device                  <value>     " << ref_device_message << std::endl;
    std::cout << "    -tolerance                   <value>     " << tolerance_message << std::endl;
    std::cout << "    -rebind                      <value>     " << rebind_message << std::endl;
    std::cout << std::endl;
    std::cout << " Model options:                              " << std::endl;
    std::cout << "    -blocks                      <value>     " << blocks_message << std::endl;
    std::cout << "    -hidden                      <value>     " << hidden_message << std::endl;
    std::cout << "    -ffn                         <value>     " << ffn_message << std::endl;
    std::cout << "    -seq                         <value>     " << seq_message << std::endl;
    std::cout << "    -ref_weights                 <value>     " << ref_weights_message << std::endl;
    std::cout << "    -seed                        <value>     " << seed_message << std::endl;
    std::cout << std::endl;
}

namespace {

constexpr auto ELEMENT_TYPE = ov::element::f16;

constexpr char MODE_UNFOLD_ASYNC[] = "unfold-async";
constexpr char MODE_UNFOLD_SYNC[] = "unfold-sync";
constexpr char MODE_EXECUTION_LIST[] = "execution-list";
constexpr char MODE_REFERENCE[] = "reference";

using Clock = std::chrono::steady_clock;

struct Dims {
    size_t hidden;
    size_t ffn;
    size_t seq;

    ov::Shape x() const {
        return {1, seq, hidden};
    }
    ov::Shape w_up() const {
        return {ffn, hidden};
    }
    ov::Shape w_down() const {
        return {hidden, ffn};
    }
};

struct Data {
    ov::Tensor input;
    std::vector<ov::Tensor> w_up;
    std::vector<ov::Tensor> w_down;
    // -rebind: two sets of block outputs, the chain alternates between them every iteration
    std::vector<std::vector<ov::Tensor>> chain;
};

std::string w_up_name(size_t block) {
    return "w_up_" + std::to_string(block);
}

std::string w_down_name(size_t block) {
    return "w_down_" + std::to_string(block);
}

bool is_npu(const std::string& device) {
    return device.rfind("NPU", 0) == 0;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(str);
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

ov::AnyMap parse_config(const std::string& config) {
    ov::AnyMap properties;
    for (const auto& entry : split(config, ',')) {
        const auto pos = entry.find('=');
        if (pos == std::string::npos || pos == 0) {
            throw std::invalid_argument("Malformed -config entry \"" + entry + "\", expected KEY=VALUE");
        }
        properties[entry.substr(0, pos)] = entry.substr(pos + 1);
    }
    return properties;
}

std::vector<std::string> parse_modes(const std::string& modes, bool execution_list_supported) {
    const std::vector<std::string> known = {MODE_UNFOLD_ASYNC, MODE_UNFOLD_SYNC, MODE_EXECUTION_LIST, MODE_REFERENCE};

    std::vector<std::string> result;
    for (const auto& mode : split(modes, ',')) {
        if (mode == "all") {
            for (const auto& name : known) {
                if (name == MODE_EXECUTION_LIST && !execution_list_supported) {
                    continue;
                }
                if (std::find(result.begin(), result.end(), name) == result.end()) {
                    result.push_back(name);
                }
            }
            continue;
        }
        if (std::find(known.begin(), known.end(), mode) == known.end()) {
            throw std::invalid_argument("Unknown mode \"" + mode + "\"");
        }
        if (std::find(result.begin(), result.end(), mode) == result.end()) {
            result.push_back(mode);
        }
    }
    if (result.empty()) {
        throw std::invalid_argument("No mode selected");
    }
    return result;
}

//
// Models
//

// An MLP block with a residual connection. The weights are inputs, the same way NPUW passes the closures of a
// repeated block to the shared compiled model.
ov::Output<ov::Node> make_block(const ov::Output<ov::Node>& x,
                                const ov::Output<ov::Node>& w_up,
                                const ov::Output<ov::Node>& w_down) {
    auto up = std::make_shared<ov::op::v0::MatMul>(x, w_up, false, true);
    auto act = std::make_shared<ov::op::v4::Swish>(up);
    auto down = std::make_shared<ov::op::v0::MatMul>(act, w_down, false, true);
    return std::make_shared<ov::op::v1::Add>(x, down);
}

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const ov::Shape& shape, const std::string& name) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ELEMENT_TYPE, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).get_tensor().set_names({name});
    return parameter;
}

std::shared_ptr<ov::Model> make_result_model(const ov::Output<ov::Node>& output,
                                             const ov::ParameterVector& parameters,
                                             const std::string& name) {
    output.get_tensor().set_names({"y"});
    auto result = std::make_shared<ov::op::v0::Result>(output);
    result->set_friendly_name("y");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, parameters, name);
}

std::shared_ptr<ov::Model> make_block_model(const Dims& dims) {
    auto x = make_parameter(dims.x(), "x");
    auto w_up = make_parameter(dims.w_up(), "w_up");
    auto w_down = make_parameter(dims.w_down(), "w_down");
    return make_result_model(make_block(x, w_up, w_down), {x, w_up, w_down}, "mlp_block");
}

std::shared_ptr<ov::Model> make_merged_model(const Dims& dims, const Data& data, bool const_weights) {
    auto x = make_parameter(dims.x(), "x");
    ov::ParameterVector parameters = {x};

    ov::Output<ov::Node> current = x;
    for (size_t block = 0; block < data.w_up.size(); ++block) {
        ov::Output<ov::Node> w_up;
        ov::Output<ov::Node> w_down;
        if (const_weights) {
            // Shares the memory of the tensors, no copy
            w_up = std::make_shared<ov::op::v0::Constant>(data.w_up[block]);
            w_down = std::make_shared<ov::op::v0::Constant>(data.w_down[block]);
        } else {
            auto w_up_parameter = make_parameter(dims.w_up(), w_up_name(block));
            auto w_down_parameter = make_parameter(dims.w_down(), w_down_name(block));
            parameters.push_back(w_up_parameter);
            parameters.push_back(w_down_parameter);
            w_up = w_up_parameter;
            w_down = w_down_parameter;
        }
        current = make_block(current, w_up, w_down);
    }

    return make_result_model(current, parameters, "mlp_merged");
}

//
// Data
//

// Allocates host tensors from the device's default context when it provides them, so the plugin can use them
// without copies. Falls back to regular tensors otherwise.
class TensorAllocator {
public:
    TensorAllocator(ov::Core& core, const std::string& device) {
        try {
            _context = core.get_default_context(device);
        } catch (const std::exception&) {
            _context = std::nullopt;
        }
    }

    ov::Tensor allocate(const ov::Shape& shape) {
        if (_context.has_value()) {
            try {
                return _context->create_host_tensor(ELEMENT_TYPE, shape);
            } catch (const std::exception&) {
                _context = std::nullopt;
            }
        }
        return ov::Tensor(ELEMENT_TYPE, shape);
    }

    bool uses_context() const {
        return _context.has_value();
    }

private:
    std::optional<ov::RemoteContext> _context;
};

void fill_uniform(ov::Tensor& tensor, float range, uint32_t seed) {
    auto* data = tensor.data<ov::float16>();
    uint32_t state = (seed * 2654435761u) | 1u;
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        // xorshift32, good enough for test data and much faster than <random> for gigabytes of weights
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float unit = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        data[i] = ov::float16((2.0f * unit - 1.0f) * range);
    }
}

Data generate_data(TensorAllocator& allocator, const Dims& dims, size_t blocks, uint32_t seed, bool rebind) {
    // Keeps the activations around unit scale: the up projection preserves the variance and the down projection is
    // scaled down, so the residual grows slowly over the blocks and stays far from the f16 range.
    const float w_up_range = std::sqrt(3.0f / static_cast<float>(dims.hidden));
    const float w_down_range = 0.1f * std::sqrt(3.0f / static_cast<float>(dims.ffn));

    Data data;
    data.input = allocator.allocate(dims.x());
    fill_uniform(data.input, 1.0f, seed);

    data.w_up.reserve(blocks);
    data.w_down.reserve(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        data.w_up.push_back(allocator.allocate(dims.w_up()));
        fill_uniform(data.w_up.back(), w_up_range, seed + 1 + static_cast<uint32_t>(2 * block));
        data.w_down.push_back(allocator.allocate(dims.w_down()));
        fill_uniform(data.w_down.back(), w_down_range, seed + 2 + static_cast<uint32_t>(2 * block));
    }

    if (rebind) {
        data.chain.resize(2);
        for (auto& outputs : data.chain) {
            for (size_t block = 0; block < blocks; ++block) {
                outputs.push_back(allocator.allocate(dims.x()));
            }
        }
    }
    return data;
}

std::vector<float> to_floats(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const ov::float16>();
    std::vector<float> result(tensor.get_size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<float>(data[i]);
    }
    return result;
}

//
// Access to implementation objects
//

// Proof of concept only. The execution list works with the dev API objects behind the public API, which the public
// API doesn't expose. A product version needs a proper public API instead, see docs/execution-list-proposal.md.

// The implementation pointer of ov::RemoteContext is protected, so a derived class can read it from a copy
class RemoteContextAccess : public ov::RemoteContext {
public:
    static std::shared_ptr<ov::IRemoteContext> get_impl(const ov::RemoteContext& context) {
        RemoteContextAccess access;
        static_cast<ov::RemoteContext&>(access) = context;
        return access._impl;
    }
};

// The implementation pointer of ov::InferRequest is private. Access checks don't apply to the template arguments of
// an explicit instantiation, which lets the instantiation below export a pointer to the member.
template <typename Tag, typename Tag::type Member>
struct PrivateMember {
    friend typename Tag::type get_member(Tag) {
        return Member;
    }
};

struct InferRequestImpl {
    using type = std::shared_ptr<ov::IAsyncInferRequest> ov::InferRequest::*;
    friend type get_member(InferRequestImpl);
};

template struct PrivateMember<InferRequestImpl, &ov::InferRequest::_impl>;

std::shared_ptr<ov::IAsyncInferRequest> get_impl(const ov::InferRequest& request) {
    return request.*get_member(InferRequestImpl{});
}

//
// Runners
//

struct Runner {
    std::string name;
    size_t submissions_per_iteration;
    std::function<void()> run_once;
    std::function<ov::Tensor()> output;
};

ov::CompiledModel compile(ov::Core& core,
                          const std::shared_ptr<ov::Model>& model,
                          const std::string& device,
                          const ov::AnyMap& config) {
    std::cout << "Compiling " << model->get_friendly_name() << " for " << device << "..." << std::endl;
    const auto start = Clock::now();
    auto compiled_model = core.compile_model(model, device, config);
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "Compiled " << model->get_friendly_name() << " in " << std::fixed << std::setprecision(1) << elapsed
              << " ms" << std::endl;
    return compiled_model;
}

// One compiled block and one infer request per block, connected the way NPUW's UnfoldInferRequest connects them:
// the weights are bound once, and every request reads the output tensor of the previous one.
struct UnfoldState {
    ov::CompiledModel compiled_model;
    std::vector<ov::InferRequest> requests;
    const Data* data = nullptr;
    size_t iteration = 0;
};

// -rebind: moves the chain to the other set of block outputs, the way NPUW rebinds the KV cache between tokens. The
// output doesn't change, but every request gets new tensor addresses before each iteration.
void rebind_chain(std::vector<ov::InferRequest>& requests, const Data& data, size_t& iteration) {
    if (data.chain.empty()) {
        return;
    }
    const auto& outputs = data.chain[iteration++ % data.chain.size()];
    for (size_t block = 0; block < requests.size(); ++block) {
        requests[block].set_tensor("x", block == 0 ? data.input : outputs[block - 1]);
        requests[block].set_tensor("y", outputs[block]);
    }
}

std::vector<ov::InferRequest> create_chained_requests(ov::CompiledModel& compiled_model, const Data& data) {
    const size_t blocks = data.w_up.size();
    std::vector<ov::InferRequest> requests;
    requests.reserve(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        auto request = compiled_model.create_infer_request();
        request.set_tensor("w_up", data.w_up[block]);
        request.set_tensor("w_down", data.w_down[block]);
        request.set_tensor("x", block == 0 ? data.input : requests.back().get_tensor("y"));
        requests.push_back(std::move(request));
    }
    return requests;
}

std::shared_ptr<UnfoldState> create_unfold_state(ov::Core& core,
                                                 const Dims& dims,
                                                 const Data& data,
                                                 const std::string& device,
                                                 ov::AnyMap config) {
    if (is_npu(device)) {
        // Set by NPUW for every NPU submodel when NPUW_UNFOLD_IREQS is enabled
        config[ov::intel_npu::run_inferences_sequentially.name()] = true;
    } else {
        std::cout << "Warning: " << device << " does not guarantee in-order execution of requests, results of "
                  << MODE_UNFOLD_ASYNC << " may be wrong" << std::endl;
    }

    auto state = std::make_shared<UnfoldState>();
    state->compiled_model = compile(core, make_block_model(dims), device, config);
    state->requests = create_chained_requests(state->compiled_model, data);
    state->data = &data;
    return state;
}

Runner create_unfold_async_runner(const std::shared_ptr<UnfoldState>& state) {
    return {MODE_UNFOLD_ASYNC,
            state->requests.size(),
            [state] {
                rebind_chain(state->requests, *state->data, state->iteration);
                for (auto& request : state->requests) {
                    request.start_async();
                }
                for (auto& request : state->requests) {
                    request.wait();
                }
            },
            [state] {
                return state->requests.back().get_tensor("y");
            }};
}

Runner create_unfold_sync_runner(const std::shared_ptr<UnfoldState>& state) {
    return {MODE_UNFOLD_SYNC,
            state->requests.size(),
            [state] {
                rebind_chain(state->requests, *state->data, state->iteration);
                for (auto& request : state->requests) {
                    request.start_async();
                    request.wait();
                }
            },
            [state] {
                return state->requests.back().get_tensor("y");
            }};
}

// The same chain of requests as the unfold modes, appended in order to one execution list. The list orders the
// requests itself, so the block model is compiled without NPU_RUN_INFERENCES_SEQUENTIALLY.
struct ExecutionListState {
    ov::CompiledModel compiled_model;
    std::vector<ov::InferRequest> requests;
    std::shared_ptr<intel_npu::IExecutionList> list;
    const Data* data = nullptr;
    size_t iteration = 0;
};

Runner create_execution_list_runner(ov::Core& core,
                                    const Dims& dims,
                                    const Data& data,
                                    const std::string& device,
                                    const ov::AnyMap& config) {
    auto state = std::make_shared<ExecutionListState>();
    state->compiled_model = compile(core, make_block_model(dims), device, config);
    state->requests = create_chained_requests(state->compiled_model, data);
    state->data = &data;

    const auto factory = std::dynamic_pointer_cast<intel_npu::IExecutionListFactory>(
        RemoteContextAccess::get_impl(state->compiled_model.get_context()));
    if (factory == nullptr) {
        throw std::runtime_error("The context of " + device + " doesn't support execution lists");
    }

    state->list = factory->create_execution_list();
    for (const auto& request : state->requests) {
        state->list->append(get_impl(request));
    }

    return {MODE_EXECUTION_LIST,
            1,
            [state] {
                // The requests are not running between iterations, so their tensors can change
                rebind_chain(state->requests, *state->data, state->iteration);
                state->list->start_async();
                state->list->wait();
            },
            [state] {
                return state->requests.back().get_tensor("y");
            }};
}

Runner create_reference_runner(ov::Core& core,
                               const Dims& dims,
                               const Data& data,
                               const std::string& device,
                               const ov::AnyMap& config,
                               bool const_weights) {
    auto compiled_model = compile(core, make_merged_model(dims, data, const_weights), device, config);
    auto request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());

    request->set_tensor("x", data.input);
    if (!const_weights) {
        for (size_t block = 0; block < data.w_up.size(); ++block) {
            request->set_tensor(w_up_name(block), data.w_up[block]);
            request->set_tensor(w_down_name(block), data.w_down[block]);
        }
    }

    return {MODE_REFERENCE,
            1,
            [request] {
                request->start_async();
                request->wait();
            },
            [request] {
                return request->get_tensor("y");
            }};
}

//
// Measurement
//

struct Stats {
    double min = 0;
    double median = 0;
    double mean = 0;
    double p90 = 0;
    double max = 0;
};

Stats compute_stats(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    Stats stats;
    stats.min = values.front();
    stats.max = values.back();
    stats.median = values[values.size() / 2];
    stats.p90 = values[std::min(values.size() - 1, (values.size() * 9) / 10)];
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    return stats;
}

Stats measure(const Runner& runner, size_t warmup, size_t iterations) {
    for (size_t i = 0; i < warmup; ++i) {
        runner.run_once();
    }

    std::vector<double> times_ms;
    times_ms.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        runner.run_once();
        times_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return compute_stats(std::move(times_ms));
}

// Returns the max-abs difference relative to the max-abs value of the reference
double relative_difference(const std::vector<float>& actual, const std::vector<float>& reference) {
    double max_diff = 0;
    double max_reference = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(actual[i] - reference[i])));
        max_reference = std::max(max_reference, static_cast<double>(std::fabs(reference[i])));
    }
    return max_diff / std::max(max_reference, 1e-6);
}

struct Result {
    std::string name;
    size_t submissions_per_iteration;
    Stats stats;
    std::optional<double> difference;
};

void print_results(const std::vector<Result>& results, size_t blocks) {
    std::optional<double> reference_median;
    for (const auto& result : results) {
        if (result.name == MODE_REFERENCE) {
            reference_median = result.stats.median;
        }
    }

    std::cout << std::endl;
    std::cout << std::left << std::setw(16) << "mode" << std::right << std::setw(8) << "submits" << std::setw(12)
              << "median ms" << std::setw(12) << "mean ms" << std::setw(12) << "min ms" << std::setw(12) << "p90 ms"
              << std::setw(12) << "max ms" << std::setw(16) << "overhead ms" << std::setw(18) << "overhead/blk us"
              << std::setw(12) << "rel diff" << std::endl;

    for (const auto& result : results) {
        std::cout << std::left << std::setw(16) << result.name << std::right << std::setw(8)
                  << result.submissions_per_iteration << std::fixed << std::setprecision(3) << std::setw(12)
                  << result.stats.median << std::setw(12) << result.stats.mean << std::setw(12) << result.stats.min
                  << std::setw(12) << result.stats.p90 << std::setw(12) << result.stats.max;

        if (reference_median.has_value() && result.name != MODE_REFERENCE) {
            const double overhead_ms = result.stats.median - *reference_median;
            std::cout << std::setw(16) << overhead_ms << std::setprecision(1) << std::setw(18)
                      << overhead_ms * 1000.0 / static_cast<double>(blocks);
        } else {
            std::cout << std::setw(16) << "-" << std::setw(18) << "-";
        }

        if (result.difference.has_value()) {
            std::cout << std::scientific << std::setprecision(2) << std::setw(12) << *result.difference;
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "overhead = median - reference median; overhead/blk = overhead divided by the number of blocks"
              << std::endl;
}

int run() {
    if (FLAGS_blocks == 0 || FLAGS_hidden == 0 || FLAGS_ffn == 0 || FLAGS_seq == 0 || FLAGS_niter == 0) {
        throw std::invalid_argument("-blocks, -hidden, -ffn, -seq and -niter must be greater than 0");
    }
    if (FLAGS_ref_weights != "param" && FLAGS_ref_weights != "const") {
        throw std::invalid_argument("-ref_weights must be \"param\" or \"const\"");
    }

    const auto modes = parse_modes(FLAGS_mode, is_npu(FLAGS_d));
    const auto config = parse_config(FLAGS_config);
    const Dims dims{FLAGS_hidden, FLAGS_ffn, FLAGS_seq};
    const size_t blocks = FLAGS_blocks;
    const bool const_weights = FLAGS_ref_weights == "const";

    const double block_weights_mb =
        static_cast<double>(2 * dims.hidden * dims.ffn * ELEMENT_TYPE.size()) / (1024.0 * 1024.0);
    std::cout << "Device:        " << FLAGS_d << std::endl;
    std::cout << "Blocks:        " << blocks << std::endl;
    std::cout << "Block:         x[1," << dims.seq << "," << dims.hidden << "] -> MatMul[" << dims.ffn << "x"
              << dims.hidden << "] -> Swish -> MatMul[" << dims.hidden << "x" << dims.ffn << "] -> Add(x)"
              << std::endl;
    std::cout << "Weights:       " << std::fixed << std::setprecision(1) << block_weights_mb << " MB per block, "
              << block_weights_mb * static_cast<double>(blocks) << " MB in total" << std::endl;
    std::cout << "Reference:     weights as " << (const_weights ? "constants" : "inputs") << std::endl;

    ov::Core core;
    TensorAllocator allocator(core, FLAGS_d);
    std::cout << "Host tensors:  " << (allocator.uses_context() ? "default context of the device" : "regular")
              << std::endl;

    std::cout << "Generating data..." << std::endl;
    const auto data = generate_data(allocator, dims, blocks, FLAGS_seed, FLAGS_rebind);
    if (FLAGS_rebind) {
        std::cout << "Rebinding:     the block requests alternate between two sets of output tensors" << std::endl;
    }

    const auto wants = [&](const char* mode) {
        return std::find(modes.begin(), modes.end(), mode) != modes.end();
    };

    // Computed once before any mode runs and released afterwards, so it doesn't change with the modes and doesn't
    // keep a compiled model alive while they are measured
    std::optional<std::vector<float>> reference_output;
    if (FLAGS_check) {
        const auto& ref_device = FLAGS_ref_device.empty() ? FLAGS_d : FLAGS_ref_device;
        std::cout << "Computing the reference output on " << ref_device << "..." << std::endl;
        // The -config properties are meant for the device under test
        const auto checker = create_reference_runner(core,
                                                     dims,
                                                     data,
                                                     ref_device,
                                                     ref_device == FLAGS_d ? config : ov::AnyMap{},
                                                     const_weights);
        checker.run_once();
        reference_output = to_floats(checker.output());
    }

    std::vector<Runner> runners;
    std::shared_ptr<UnfoldState> unfold_state;
    if (wants(MODE_UNFOLD_ASYNC) || wants(MODE_UNFOLD_SYNC)) {
        unfold_state = create_unfold_state(core, dims, data, FLAGS_d, config);
    }
    for (const auto& mode : modes) {
        if (mode == MODE_UNFOLD_ASYNC) {
            runners.push_back(create_unfold_async_runner(unfold_state));
        } else if (mode == MODE_UNFOLD_SYNC) {
            runners.push_back(create_unfold_sync_runner(unfold_state));
        } else if (mode == MODE_EXECUTION_LIST) {
            runners.push_back(create_execution_list_runner(core, dims, data, FLAGS_d, config));
        } else if (mode == MODE_REFERENCE) {
            runners.push_back(create_reference_runner(core, dims, data, FLAGS_d, config, const_weights));
        }
    }

    bool check_failed = false;
    std::vector<Result> results;
    for (const auto& runner : runners) {
        std::cout << "Running " << runner.name << "..." << std::endl;
        const auto stats = measure(runner, FLAGS_warmup, FLAGS_niter);

        // Every iteration computes the same output from the same input, so the output left by the last iteration
        // must still match the reference
        std::optional<double> difference;
        if (reference_output.has_value()) {
            difference = relative_difference(to_floats(runner.output()), *reference_output);
            if (*difference > FLAGS_tolerance) {
                std::cout << "Output of " << runner.name << " after " << FLAGS_warmup + FLAGS_niter
                          << " iterations differs from the reference: relative difference " << *difference
                          << " > tolerance " << FLAGS_tolerance << std::endl;
                check_failed = true;
            }
        }

        results.push_back({runner.name, runner.submissions_per_iteration, stats, difference});
    }

    print_results(results, blocks);

    if (check_failed) {
        std::cout << "Check FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    if (reference_output.has_value()) {
        std::cout << "Check PASSED" << std::endl;
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h) {
            showUsage();
            return EXIT_SUCCESS;
        }
        return run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
