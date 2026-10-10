// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Batched greedy generation for the 27B (qwen3_5) paged export, for running an
// evaluation harness against the GPU plugin. Requests are read from a file and
// served with continuous batching: up to `batch` sequences decode together, and
// a finished slot is refilled (one prefill, alone) from the pending queue.
//
//   paged_serve_llm_27b <lm.xml> <embeddings.xml> <device> <requests> <outputs> [batch]
//
// <requests>: one line per request, `max_new_tokens;id,id,id,...`
// <outputs> : one line per request in the same order, the generated ids
//             (including the EOS that stopped it, if any).
// Env: BENCH_MAX_LEN (per-sequence context, default 6144), BENCH_PRECISION,
//      BENCH_EOS (comma list, default 248044,248046), BENCH_BLOCK_SIZE,
//      BENCH_VERBOSE=1 (per-request progress on stderr),
//      BENCH_TIMINGS=<csv> (per request: prompt and generated tokens, TTFT, decode time),
//      BENCH_PROFILE=1 (device time per op type, summed over prefill and decode steps).

#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
// dev API: ROI copy into a remote tensor (zeroing one slot of a state table)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <openvino/runtime/iremote_tensor.hpp>
#include <openvino/runtime/make_tensor.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kFullAttnLayers = 16;
constexpr int32_t kLinearAttnLayers = 48;
constexpr int32_t kHidden = 5120;
constexpr int32_t kLinearAttnSlots = 2;  // read + write slot per sequence

std::vector<int64_t> parse_ids(const std::string& csv) {
    std::vector<int64_t> ids;
    std::stringstream stream(csv);
    std::string value;
    while (std::getline(stream, value, ','))
        if (!value.empty())
            ids.push_back(std::stoll(value));
    return ids;
}

void set_i32(ov::InferRequest& request, const std::string& name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void set_i32_scalar(ov::InferRequest& request, const std::string& name, int32_t value) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{});
    *tensor.data<int32_t>() = value;
    request.set_tensor(name, tensor);
}

struct Request {
    int max_new = 256;
    std::vector<int64_t> prompt;
};

struct Slot {
    bool active = false;
    size_t req = 0;
    int32_t past = 0;       // tokens already in the caches
    int32_t read_slot = 0;  // linear-attention state slot to read this step
    int64_t next = 0;       // token to feed this step
    std::vector<int64_t> generated;
    std::chrono::high_resolution_clock::time_point t_start, t_first;
};

struct Timing {
    double ttft_ms = 0, decode_ms = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: paged_serve_llm_27b <lm.xml> <embeddings.xml> <device> <requests> <outputs> [batch]\n";
        return 1;
    }
    const std::string lm_path = argv[1], embed_path = argv[2], device = argv[3];
    const std::string req_path = argv[4], out_path = argv[5];
    const int32_t batch = argc > 6 ? std::stoi(argv[6]) : 8;
    const int32_t block_size = std::getenv("BENCH_BLOCK_SIZE") ? std::stoi(std::getenv("BENCH_BLOCK_SIZE")) : 16;
    const int32_t max_context_len = std::getenv("BENCH_MAX_LEN") ? std::stoi(std::getenv("BENCH_MAX_LEN")) : 6144;
    const int32_t blocks_per_seq = (max_context_len + block_size - 1) / block_size;
    const auto eos = parse_ids(std::getenv("BENCH_EOS") ? std::getenv("BENCH_EOS") : "248044,248046");
    const bool verbose = std::getenv("BENCH_VERBOSE") != nullptr;

    std::vector<Request> requests;
    {
        std::ifstream in(req_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty())
                continue;
            const auto semi = line.find(';');
            Request r;
            r.max_new = std::stoi(line.substr(0, semi));
            r.prompt = parse_ids(line.substr(semi + 1));
            if (r.prompt.empty() || static_cast<int32_t>(r.prompt.size()) + r.max_new > max_context_len) {
                std::cerr << "request " << requests.size() << ": " << r.prompt.size() << "+" << r.max_new
                          << " tokens do not fit BENCH_MAX_LEN=" << max_context_len << "\n";
                return 1;
            }
            requests.push_back(std::move(r));
        }
    }
    std::vector<std::vector<int64_t>> outputs(requests.size());
    std::vector<Timing> timings(requests.size());
    std::cerr << requests.size() << " requests, batch " << batch << ", context " << max_context_len << "\n";

    try {
        ov::Core core;
        const char* precision = std::getenv("BENCH_PRECISION");
        ov::AnyMap config{{"INFERENCE_PRECISION_HINT", precision ? precision : "f16"}};
        const bool profile = std::getenv("BENCH_PROFILE") != nullptr;
        if (profile)
            config[ov::enable_profiling.name()] = true;
        if (const char* cache = std::getenv("OV_CACHE_DIR"))
            core.set_property(ov::cache_dir(cache));

        const auto t_compile = std::chrono::high_resolution_clock::now();
        auto embed_model = core.read_model(embed_path);
        embed_model->reshape({{embed_model->input(0).get_any_name(), ov::PartialShape{1, -1}}});
        auto embed = core.compile_model(embed_model, device, config);
        auto lm_model = core.read_model(lm_path);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        manager.run_passes(lm_model);
        // Only the last token of a prefill needs logits, but the graph computes
        // them for every row (1260 x 248320 fp32 = 1.25 GB copied back). Gather
        // the wanted rows in front of the head; decode asks for every row.
        {
            std::shared_ptr<ov::Node> head_in;
            const std::string suffix = "language_model.norm/aten::mul/Multiply_1";
            for (const auto& node : lm_model->get_ordered_ops()) {
                const auto& n = node->get_friendly_name();
                if (n.size() >= suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
                    head_in = node;
            }
            if (!head_in)
                throw std::runtime_error("final norm not found; cannot insert the logit-row gather");
            // the token axis is the dynamic one ([tokens, 1, hidden] on the paged path)
            const auto& pshape = head_in->get_output_partial_shape(0);
            int64_t token_axis = -1;
            for (int64_t i = 0; i < pshape.rank().get_length(); ++i)
                if (pshape[i].is_dynamic()) {
                    token_axis = i;
                    break;
                }
            if (token_axis < 0)
                throw std::runtime_error("no dynamic token axis on the head input " + pshape.to_string());
            // snapshot before the Gather becomes a consumer itself
            const auto consumers = head_in->output(0).get_target_inputs();
            auto rows = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
            rows->set_friendly_name("logit_rows");
            rows->output(0).set_names({"logit_rows"});
            auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {token_axis});
            auto gather = std::make_shared<ov::op::v8::Gather>(head_in->output(0), rows, axis);
            for (auto target : consumers)
                target.replace_source_output(gather->output(0));
            lm_model->add_parameters({rows});
            lm_model->validate_nodes_and_infer_types();
        }
        auto lm = core.compile_model(lm_model, device, config);
        std::cerr << "compile " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_compile).count() << " s\n";

        auto embed_request = embed.create_infer_request();
        auto request = lm.create_infer_request();

        std::vector<ov::Tensor> caches;
        std::vector<std::string> cache_names;
        ov::RemoteContext context = lm.get_context();
        auto bind = [&](const std::string& name, size_t blocks) {
            const auto port = lm.input(name);
            const auto partial = port.get_partial_shape();
            ov::Shape shape(partial.size());
            for (size_t i = 0; i < partial.size(); ++i)
                shape[i] = partial[i].is_dynamic() ? blocks : static_cast<size_t>(partial[i].get_length());
            ov::Tensor zeros(port.get_element_type(), shape);
            std::memset(zeros.data(), 0, zeros.get_byte_size());
            auto remote = context.create_tensor(port.get_element_type(), shape);
            remote.copy_from(zeros);
            caches.emplace_back(std::move(remote));
            cache_names.push_back(name);
            request.set_tensor(name, caches.back());
        };
        for (int32_t layer = 0; layer < kFullAttnLayers; ++layer) {
            bind("key_cache." + std::to_string(layer), static_cast<size_t>(blocks_per_seq) * batch);
            bind("value_cache." + std::to_string(layer), static_cast<size_t>(blocks_per_seq) * batch);
        }
        for (int32_t layer = 0; layer < kLinearAttnLayers; ++layer) {
            bind("conv_state_table." + std::to_string(layer), static_cast<size_t>(kLinearAttnSlots) * batch);
            bind("gated_delta_state_table." + std::to_string(layer), static_cast<size_t>(kLinearAttnSlots) * batch);
        }
        set_i32_scalar(request, "max_context_len", max_context_len);
        // One zeroed device tensor per state-table shape ([2 slots, ...]);
        // zero_slot_state copies it over a slot pair device-to-device.
        std::vector<size_t> la_cache_idx;
        std::map<std::string, ov::Tensor> zero_blocks;
        for (size_t i = 0; i < caches.size(); ++i) {
            if (cache_names[i].rfind("key_cache", 0) == 0 || cache_names[i].rfind("value_cache", 0) == 0)
                continue;
            la_cache_idx.push_back(i);
            ov::Shape slot_shape = caches[i].get_shape();
            slot_shape[0] = kLinearAttnSlots;
            const std::string key = caches[i].get_element_type().get_type_name() + ":" + slot_shape.to_string();
            if (!zero_blocks.count(key)) {
                ov::Tensor zeros(caches[i].get_element_type(), slot_shape);
                std::memset(zeros.data(), 0, zeros.get_byte_size());
                auto remote = context.create_tensor(caches[i].get_element_type(), slot_shape);
                remote.copy_from(zeros);
                zero_blocks.emplace(key, std::move(remote));
            }
        }
        auto zero_slot_state = [&](int32_t s) {
            for (const size_t i : la_cache_idx) {
                ov::Shape slot_shape = caches[i].get_shape();
                slot_shape[0] = kLinearAttnSlots;
                const std::string key = caches[i].get_element_type().get_type_name() + ":" + slot_shape.to_string();
                const size_t slot_bytes = caches[i].get_byte_size() / caches[i].get_shape()[0];
                auto dst = std::dynamic_pointer_cast<ov::IRemoteTensor>(ov::get_tensor_impl(caches[i])._ptr);
                auto src = ov::get_tensor_impl(zero_blocks.at(key))._ptr;
                if (!dst)
                    throw std::runtime_error("state table is not a remote tensor");
                dst->copy_from(src, 0, static_cast<size_t>(s) * kLinearAttnSlots * slot_bytes, slot_shape);
            }
        };
        {
            size_t bytes = 0;
            for (const auto& t : caches)
                bytes += t.get_byte_size();
            std::cerr << "caches: " << bytes / (1024 * 1024) << " MiB\n";
        }

        const auto embeds_type = lm.input("inputs_embeds").get_element_type();
        auto feed_embeds = [&](const std::vector<int64_t>& ids) {
            ov::Tensor input(ov::element::i64, ov::Shape{1, ids.size()});
            std::copy(ids.begin(), ids.end(), input.data<int64_t>());
            embed_request.set_input_tensor(input);
            embed_request.infer();
            const auto out = embed_request.get_output_tensor();
            ov::Tensor embeds(embeds_type, ov::Shape{ids.size(), kHidden});
            std::memcpy(embeds.data(), out.data(), embeds.get_byte_size());
            request.set_tensor("inputs_embeds", embeds);
        };

        // One infer over the given (slot, token count) pairs; tokens of each
        // slot start at its `past`. Fills the paged-attention and linear-
        // attention metadata for every subsequence in the batch.
        std::vector<Slot> slots(batch);
        // [0] prefill, [1] decode: op type -> (us, count)
        std::map<std::string, std::pair<double, size_t>> prof[2];
        size_t prof_steps[2] = {0, 0};
        // The plugin reports each node's mean over all infers so far; mean times
        // infer count is its running total, and the step's share is the change.
        std::map<std::string, double> prof_total;
        size_t infers = 0;
        auto run_step = [&](const std::vector<std::pair<int32_t, int32_t>>& parts) {
            size_t total = 0;
            for (const auto& p : parts)
                total += static_cast<size_t>(p.second);
            ov::Tensor positions(ov::element::i64, ov::Shape{4, total});
            auto* pos = positions.data<int64_t>();
            std::vector<int32_t> past_lens, subseq{0}, blocks, block_begins{0}, la_past, la_blocks, la_begins{0};
            size_t tok = 0;
            for (const auto& [s, len] : parts) {
                const Slot& sl = slots[s];
                for (int32_t i = 0; i < len; ++i, ++tok)
                    for (size_t section = 0; section < 4; ++section)
                        pos[section * total + tok] = sl.past + i;
                past_lens.push_back(sl.past);
                subseq.push_back(subseq.back() + len);
                const int32_t active = (sl.past + len + block_size - 1) / block_size;
                for (int32_t b = 0; b < active; ++b)
                    blocks.push_back(s * blocks_per_seq + b);
                block_begins.push_back(block_begins.back() + active);
                la_past.push_back(sl.past);
                la_blocks.push_back(s * kLinearAttnSlots + sl.read_slot);
                la_blocks.push_back(s * kLinearAttnSlots + (1 - sl.read_slot));
                la_begins.push_back(la_begins.back() + kLinearAttnSlots);
            }
            request.set_tensor("position_ids", positions);
            set_i32(request, "past_lens", past_lens);
            set_i32(request, "subsequence_begins", subseq);
            set_i32(request, "block_indices", blocks);
            set_i32(request, "block_indices_begins", block_begins);
            set_i32(request, "la.past_lens", la_past);
            set_i32(request, "la.block_indices", la_blocks);
            set_i32(request, "la.block_indices_begins", la_begins);
            set_i32(request, "la.cache_interval", std::vector<int32_t>(parts.size(), 0));
            // last token of each subsequence
            std::vector<int32_t> rows;
            for (size_t i = 1; i < subseq.size(); ++i)
                rows.push_back(subseq[i] - 1);
            set_i32(request, "logit_rows", rows);
            request.infer();
            if (profile) {
                const int phase = total == parts.size() ? 1 : 0;
                ++prof_steps[phase];
                ++infers;
                for (const auto& p : request.get_profiling_info()) {
                    if (p.status == ov::ProfilingInfo::Status::NOT_RUN)
                        continue;
                    const double running = static_cast<double>(p.real_time.count()) * static_cast<double>(infers);
                    auto& last = prof_total[p.node_name];
                    auto& e = prof[phase][p.node_type];
                    e.first += running - last;
                    e.second += 1;
                    last = running;
                }
            }
            for (const auto& [s, len] : parts) {
                slots[s].past += len;
                slots[s].read_slot = 1 - slots[s].read_slot;
            }
        };
        auto argmax_row = [&](const ov::Tensor& logits, size_t row) {
            const auto shape = logits.get_shape();
            const size_t vocab = shape.back();
            const float* r = logits.data<float>() + row * vocab;
            return static_cast<int64_t>(std::distance(r, std::max_element(r, r + vocab)));
        };
        auto is_eos = [&](int64_t t) {
            return std::find(eos.begin(), eos.end(), t) != eos.end();
        };

        std::deque<size_t> pending;
        for (size_t i = 0; i < requests.size(); ++i)
            pending.push_back(i);
        size_t done = 0, steps = 0, decoded_tokens = 0;
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto finish = [&](int32_t s) {
            using ms = std::chrono::duration<double, std::milli>;
            outputs[slots[s].req] = slots[s].generated;
            timings[slots[s].req] = {ms(slots[s].t_first - slots[s].t_start).count(), ms(std::chrono::high_resolution_clock::now() - slots[s].t_first).count()};
            slots[s].active = false;
            ++done;
            if (verbose)
                std::cerr << "  done " << done << "/" << requests.size() << " req " << slots[s].req << " gen " << slots[s].generated.size() << "\n";
        };

        while (done < requests.size()) {
            // refill free slots: prefill alone, take the first token
            for (int32_t s = 0; s < batch && !pending.empty(); ++s) {
                if (slots[s].active)
                    continue;
                const size_t r = pending.front();
                pending.pop_front();
                slots[s] = Slot{};
                // The paged linear-attention ops always start from the state in
                // block_indices[begin] (past_len == 0 does not imply zero), so a
                // reused slot must have its two state blocks cleared first.
                zero_slot_state(s);
                slots[s].active = true;
                slots[s].req = r;
                slots[s].t_start = std::chrono::high_resolution_clock::now();
                feed_embeds(requests[r].prompt);
                run_step({{s, static_cast<int32_t>(requests[r].prompt.size())}});
                const auto logits = request.get_output_tensor();
                const size_t rows = logits.get_size() / logits.get_shape().back();
                slots[s].next = argmax_row(logits, rows - 1);
                slots[s].t_first = std::chrono::high_resolution_clock::now();
                slots[s].generated.push_back(slots[s].next);
                if (is_eos(slots[s].next) || requests[r].max_new <= 1)
                    finish(s);
            }
            // joint decode step over the active slots
            std::vector<int32_t> active;
            std::vector<int64_t> tokens;
            std::vector<std::pair<int32_t, int32_t>> parts;
            for (int32_t s = 0; s < batch; ++s) {
                if (!slots[s].active)
                    continue;
                active.push_back(s);
                tokens.push_back(slots[s].next);
                parts.emplace_back(s, 1);
            }
            if (active.empty())
                continue;  // every refilled slot finished on its first token
            feed_embeds(tokens);
            run_step(parts);
            const auto logits = request.get_output_tensor();
            const size_t rows = logits.get_size() / logits.get_shape().back();
            if (rows != active.size()) {
                std::cerr << "logits rows " << rows << " != active " << active.size() << " shape " << logits.get_shape() << "\n";
                return 1;
            }
            ++steps;
            for (size_t i = 0; i < active.size(); ++i) {
                Slot& sl = slots[active[i]];
                sl.next = argmax_row(logits, i);
                sl.generated.push_back(sl.next);
                ++decoded_tokens;
                const auto& req = requests[sl.req];
                if (is_eos(sl.next) || static_cast<int>(sl.generated.size()) >= req.max_new || sl.past + 1 >= max_context_len)
                    finish(active[i]);
            }
            if (verbose && steps % 200 == 0)
                std::cerr << "  step " << steps << " active " << active.size() << " done " << done << "\n";
        }
        const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        std::cerr << "served " << requests.size() << " requests, " << decoded_tokens << " decode tokens in " << steps << " steps, " << secs << " s ("
                  << decoded_tokens / secs << " tok/s aggregate)\n";

        std::ofstream out(out_path);
        for (const auto& ids : outputs) {
            for (size_t i = 0; i < ids.size(); ++i)
                out << (i ? "," : "") << ids[i];
            out << '\n';
        }
        if (const char* tpath = std::getenv("BENCH_TIMINGS")) {
            // With batch > 1 the decode time also covers other slots' prefills and steps.
            std::ofstream tout(tpath);
            tout << "request,prompt_tokens,generated_tokens,ttft_ms,decode_ms\n";
            for (size_t i = 0; i < requests.size(); ++i)
                tout << i << ',' << requests[i].prompt.size() << ',' << outputs[i].size() << ',' << timings[i].ttft_ms << ',' << timings[i].decode_ms << '\n';
        }
        for (int phase = 0; profile && phase < 2; ++phase) {
            std::vector<std::pair<std::string, std::pair<double, size_t>>> v(prof[phase].begin(), prof[phase].end());
            std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
                return a.second.first > b.second.first;
            });
            double total = 0;
            for (const auto& e : v)
                total += e.second.first;
            std::cerr << (phase ? "decode" : "prefill") << " profile: " << prof_steps[phase] << " steps, " << total / 1000.0 << " ms device time\n";
            for (const auto& e : v)
                std::cerr << "  " << std::setw(32) << std::left << e.first << std::setw(12) << std::right << e.second.first / 1000.0
                          << " ms  n=" << e.second.second << "  (" << 100.0 * e.second.first / total << "%)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paged 27b serve failed: " << error.what() << '\n';
        return 1;
    }
}
