// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Internal: the backend/buffer machinery around the graph.
//
// A GgmlModel is built from a dumped llama.cpp cgraph artifact (cgraph_loader.cpp). Everything
// AROUND the graph -- backend selection, weight upload, the external context that keeps inputs
// and KV caches alive across compute() calls, graph allocation -- lives here so the loader only
// has to describe the topology. The public GgmlModel surface is implemented in model.cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include "openvino/core/except.hpp"
#include "openvino/ggml_cgraph_loader/ggml_model.hpp"

namespace ov {
namespace ggml_cgraph_loader {

constexpr int MAX_NODES = 8192;

/// fseek to a 64-bit offset (long is 32-bit on Windows, so fseek fails past 2 GB).
inline int seek64(FILE* f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<__int64>(offset), SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
}

/// Graph outputs, owned by the caller: the producer (the Loader) is destroyed after building, so
/// anything needed afterwards must live here.
struct Out {
    std::map<std::string, ggml_tensor*> map, externals, named_outputs;
    std::set<std::string> unsupported;
    ggml_tensor* last = nullptr;
    /// Persistent per-sequence state buffers: recurrent (Mamba / RWKV) state, whatever the model names it.
    /// KV caches are excluded; they need no reset because slots past the sequence are masked.
    std::set<std::string> state_names;
    /// Every node, in topological order. Needed because a forward expansion from the logits
    /// alone MISSES the KV cache writes: SET_ROWS results are not reachable from the output, so
    /// they would be dropped from the graph and attention would read an empty cache. Expanding
    /// in this order also reproduces the original node ordering, which is what makes the cache
    /// write land before the read.
    std::vector<ggml_tensor*> order;
};

struct GgmlModel::Impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, ebuf = nullptr;
    ggml_context *wctx = nullptr, *ctx_g = nullptr, *ctx_ext = nullptr;
    gguf_context* gg = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    std::map<std::string, ggml_tensor*> wmap;  // gguf tensor name -> uploaded weight
    size_t n_kv = 0;  // baked into the graph shapes; 0 when unknown
    /// Inputs whose value the artifact carries; written once after allocation.
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> constants;
    Out out;
    /// Set when built over another model (from_cgraph(json, base)): backend, weights, gguf and the
    /// shared caches belong to it, and this keeps it alive.
    std::shared_ptr<GgmlModel> base;
    std::set<ggml_tensor*> shared_caches;
    bool alibi = false;  // some attention node has max_bias > 0
    /// Set when some op cannot run on `backend` (e.g. a kernel the GPU backend lacks for these
    /// shapes): the scheduler then runs just those ops on the CPU backend, as llama.cpp does.
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<std::string> fallback_ops;

    ~Impl() {
        if (sched) ggml_backend_sched_free(sched);
        if (cpu) ggml_backend_free(cpu);
        if (alloc) ggml_gallocr_free(alloc);
        if (ebuf) ggml_backend_buffer_free(ebuf);
        if (ctx_ext) ggml_free(ctx_ext);
        if (ctx_g) ggml_free(ctx_g);
        if (base) {
            return;
        }
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (gg) gguf_free(gg);
        if (backend) ggml_backend_free(backend);
    }

    /// Take over `other`'s backend, weights and gguf metadata instead of loading them again.
    void share_from(const std::shared_ptr<GgmlModel>& other, const Impl& o) {
        base = other;
        backend = o.backend;
        wctx = o.wctx;
        gg = o.gg;
        wmap = o.wmap;
    }

    /// Pick and initialise a ggml backend. Empty name means GPU, then IGPU (Intel integrated
    /// graphics register as IGPU, not GPU), then CPU.
    void init_backend(const std::string& backend_name) {
        ggml_backend_load_all();
        ggml_backend_dev_t dev = nullptr;
        if (!backend_name.empty()) {
            dev = ggml_backend_dev_by_name(backend_name.c_str());
            OPENVINO_ASSERT(dev, "[GGML] no such ggml backend device: ", backend_name);
        } else {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
            if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        OPENVINO_ASSERT(dev, "[GGML] no ggml backend device available");
        backend = ggml_backend_dev_init(dev, nullptr);
        OPENVINO_ASSERT(backend, "[GGML] failed to initialise backend");
    }

    /// Read every tensor out of the .gguf and upload it to the backend, keyed by gguf name.
    void load_weights(const std::string& gguf_path) {
        gguf_init_params gp = {true, &wctx};
        gg = gguf_init_from_file(gguf_path.c_str(), gp);
        OPENVINO_ASSERT(gg, "[GGML] cannot read gguf: ", gguf_path);
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        FILE* f = fopen(gguf_path.c_str(), "rb");
        OPENVINO_ASSERT(f, "[GGML] cannot open: ", gguf_path);
        const size_t doff = gguf_get_data_offset(gg);
        std::vector<uint8_t> tmp;
        for (ggml_tensor* t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            const int i = gguf_find_tensor(gg, ggml_get_name(t));
            if (i < 0) {
                continue;
            }
            tmp.resize(ggml_nbytes(t));
            if (seek64(f, doff + gguf_get_tensor_offset(gg, i)) != 0 ||
                fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) {
                fclose(f);
                OPENVINO_THROW("[GGML] short read for tensor ", ggml_get_name(t));
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
            wmap[ggml_get_name(t)] = t;
        }
        fclose(f);
    }

    void init_contexts() {
        ctx_ext = ggml_init({ggml_tensor_overhead() * 1024, nullptr, true});
        ctx_g = ggml_init({ggml_tensor_overhead() * MAX_NODES +
                               ggml_graph_overhead_custom(MAX_NODES, false),
                           nullptr, true});
    }

    /// Allocate the external tensors, expand the forward graph, allocate it, and zero the KV
    /// caches. Shared tail of the build path.
    void finalize() {
        OPENVINO_ASSERT(out.last, "[GGML] no output tensor produced");
        // Every external may be shared with the base model, leaving nothing to allocate here.
        if (ggml_get_first_tensor(ctx_ext)) {
            ebuf = ggml_backend_alloc_ctx_tensors(ctx_ext, backend);
        }
        gf = ggml_new_graph_custom(ctx_g, MAX_NODES, false);
        if (out.order.empty()) {
            ggml_build_forward_expand(gf, out.last);
        } else {
            for (ggml_tensor* t : out.order) {
                ggml_build_forward_expand(gf, t);  // already-visited nodes are skipped
            }
        }
        // Ops the backend cannot run in these shapes and types go to the CPU backend through the
        // scheduler; with none (the common case) the graph runs on one backend directly.
        std::set<std::string> unsupported_here;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor* t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_NONE && !ggml_backend_supports_op(backend, t)) {
                unsupported_here.insert(ggml_op_desc(t));
            }
        }
        if (!unsupported_here.empty()) {
            const bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
            std::string list;
            for (const auto& u : unsupported_here) {
                list += (list.empty() ? "" : ", ") + u;
            }
            OPENVINO_ASSERT(!is_cpu, "[GGML] the CPU backend cannot run: ", list);
            cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            OPENVINO_ASSERT(cpu, "[GGML] no CPU backend to run ", list, " on");
            ggml_backend_t backends[2] = {backend, cpu};
            sched = ggml_backend_sched_new(backends, nullptr, 2, MAX_NODES, false, false);
            OPENVINO_ASSERT(ggml_backend_sched_alloc_graph(sched, gf), "[GGML] graph allocation failed");
            fallback_ops.assign(unsupported_here.begin(), unsupported_here.end());
        } else {
            alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            OPENVINO_ASSERT(ggml_gallocr_alloc_graph(alloc, gf), "[GGML] graph allocation failed");
        }

        for (const auto& c : constants) {
            ggml_backend_tensor_set(c.first, c.second.data(), 0, c.second.size());
        }

        // KV caches live in ctx_ext, so they are allocated once and persist across compute()
        // calls; SET_ROWS writes through a view into that same buffer.
        for (const auto& kv : out.externals) {
            if ((kv.first.rfind("cache_", 0) != 0 && !out.state_names.count(kv.first)) || shared_caches.count(kv.second)) {
                continue;
            }
            std::vector<char> z(ggml_nbytes(kv.second), 0);
            ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
        }
    }

    void throw_if_unsupported() const {
        if (out.unsupported.empty()) {
            return;
        }
        std::string list;
        for (const auto& u : out.unsupported) {
            list += (list.empty() ? "" : ", ") + u;
        }
        OPENVINO_THROW("[GGML] cannot translate: ", list);
    }
};

}  // namespace ggml_cgraph_loader
}  // namespace ov
