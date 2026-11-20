// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/reorder.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu {
namespace fc_decode_cache {
struct Budget {
    std::mutex mutex;
    size_t limit = 0;
    size_t used = 0;
    size_t peak = 0;
    bool trace = false;
    std::set<std::string> selected;
    explicit Budget() {
        if (const char* value = std::getenv("OV_FC_DECODE_CACHE_MIB")) {
            const auto mib = std::stoull(value);
            OPENVINO_ASSERT(mib <= 4096);
            limit = mib * 1024 * 1024;
        }
        trace = std::getenv("OV_FC_DECODE_CACHE_TRACE") != nullptr;
        if (const char* path = std::getenv("OV_FC_DECODE_CACHE_NODES")) {
            std::ifstream input(path);
            OPENVINO_ASSERT(input.good(), "Missing decode cache selection file");
            std::string line;
            while (std::getline(input, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (!line.empty())
                    selected.insert(line);
            }
        }
    }
    bool reserve(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > limit - used)
            return false;
        used += bytes;
        peak = std::max(peak, used);
        return true;
    }
    void release(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        OPENVINO_ASSERT(bytes <= used);
        used -= bytes;
    }
    ~Budget() {
        if (trace)
            std::fprintf(stderr, "[FC-CACHE] BUDGET limit=%zu peak=%zu released=%d\n", limit, peak, used == 0);
    }
};
inline std::shared_ptr<Budget> model_budget(const GraphContext::CPtr& context) {
    static std::mutex registryMutex;
    static std::unordered_map<const GraphContext*, std::weak_ptr<Budget>> registry;
    std::lock_guard<std::mutex> lock(registryMutex);
    for (auto it = registry.begin(); it != registry.end();) {
        if (it->second.expired())
            it = registry.erase(it);
        else
            ++it;
    }
    auto& slot = registry[context.get()];
    if (auto current = slot.lock())
        return current;
    auto result = std::make_shared<Budget>();
    slot = result;
    return result;
}
inline bool plain(const MemoryPtr& memory) {
    auto md = MemoryDescUtils::convertToDnnlMemoryDesc(memory->getDescPtr())->getDnnlDesc();
    if (md.get_format_kind() != dnnl::memory::format_kind::blocked || md.get_inner_nblks() != 0 ||
        md.get_submemory_offset() != 0)
        return false;
    auto dims = md.get_dims();
    auto strides = md.get_strides();
    if (dims != md.get_padded_dims())
        return false;
    for (auto offset : md.get_padded_offsets())
        if (offset != 0)
            return false;
    int64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        if (strides[i] != stride)
            return false;
        stride *= dims[i];
    }
    return true;
}
inline float read(const MemoryPtr& mem, size_t index) {
    const auto type = mem->getPrecision();
    const auto* bytes = static_cast<const uint8_t*>(mem->getData());
    if (type == ov::element::f16)
        return static_cast<const ov::float16*>(mem->getData())[index];
    if (type == ov::element::f32)
        return static_cast<const float*>(mem->getData())[index];
    if (type == ov::element::i32)
        return static_cast<const int32_t*>(mem->getData())[index];
    if (type == ov::element::i8)
        return static_cast<const int8_t*>(mem->getData())[index];
    if (type == ov::element::u8)
        return bytes[index];
    if (type == ov::element::i4 || type == ov::element::u4) {
        int value = (bytes[index / 2] >> (4 * (index % 2))) & 15;
        if (type == ov::element::i4 && value >= 8)
            value -= 16;
        return static_cast<float>(value);
    }
    OPENVINO_THROW("Unsupported decode-cache dequantization precision ", type);
}
struct Params {
    MemoryPtr memory;
    size_t channels = 1, groups = 1;
    explicit Params(const MemoryPtr& value, size_t n, size_t k) : memory(value) {
        OPENVINO_ASSERT(plain(value));
        auto dims = value->getShape().getStaticDims();
        while (dims.size() > 2 && dims.front() == 1)
            dims.erase(dims.begin());
        if (dims.size() == 1 && dims.front() == 1)
            dims.push_back(1);
        OPENVINO_ASSERT(dims.size() == 2 && dims[0] > 0 && dims[1] > 0);
        channels = dims[0];
        groups = dims[1];
        OPENVINO_ASSERT(n % channels == 0 && k % groups == 0);
    }
    float at(size_t nc, size_t kc, size_t n, size_t k) const {
        return read(memory, (nc / (n / channels)) * groups + kc / (k / groups));
    }
};
class DecodeCache : public Executor {
    ExecutorPtr original;
    FCAttrs fcAttrs;
    ExecutorContext::CPtr context;
    std::shared_ptr<Budget> budget;
    std::string name;
    DnnlMatMulPrimitivePtr primitive;
    DnnlShapeAgnosticDataPtr shapeData;
    MemoryPtr weights, scratch;
    dnnl_primitive_args args;
    MemoryDescPtr srcDesc, dstDesc;
    size_t charged = 0;
    size_t packedBytes = 0;
    size_t decodeCalls = 0, compressedCalls = 0;
    bool failed = false, active = false;
    int cachedNuma = -1;
    bool tracedUpdate = false;
    void reset_cache() {
        args.clear();
        weights.reset();
        scratch.reset();
        primitive.reset();
        shapeData.reset();
        srcDesc.reset();
        dstDesc.reset();
        active = false;
        cachedNuma = -1;
        if (charged) {
            budget->release(charged);
            charged = 0;
        }
    }
    void initialize(const MemoryArgs& memory) {
        const auto quant = memory.at(ARG_WEI);
        const auto type = quant->getPrecision();
        OPENVINO_ASSERT(fcAttrs.constantWeights && !fcAttrs.weightsNonTransposed && !fcAttrs.sparseWeights);
        OPENVINO_ASSERT(fcAttrs.postOps.empty(), "Decode cache excludes fused postops");
        OPENVINO_ASSERT(type == ov::element::i4 || type == ov::element::u4 || type == ov::element::i8 ||
                        type == ov::element::u8);
        OPENVINO_ASSERT(plain(quant));
        auto dims = quant->getShape().getStaticDims();
        OPENVINO_ASSERT(dims.size() == 2);
        const size_t n = dims[0], k = dims[1];
        OPENVINO_ASSERT(memory.at(ARG_SRC)->getPrecision() == ov::element::f16);
        OPENVINO_ASSERT(memory.at(ARG_SRC)->getShape().getStaticDims().back() == k);
        OPENVINO_ASSERT(memory.at(ARG_DST)->getShape().getStaticDims().back() == n);
        OPENVINO_ASSERT(plain(memory.at(ARG_SRC)) && plain(memory.at(ARG_DST)));
        Params scales(memory.at(ARG_WEI | ARG_ATTR_SCALES), n, k);
        std::unique_ptr<Params> zp;
        if (auto it = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS); it != memory.end()) {
            zp = std::make_unique<Params>(it->second, n, k);
            const float minimum = type == ov::element::i4 ? -8.F : type == ov::element::i8 ? -128.F : 0.F;
            const float maximum = type == ov::element::i4   ? 7.F
                                  : type == ov::element::u4 ? 15.F
                                  : type == ov::element::i8 ? 127.F
                                                            : 255.F;
            for (size_t i = 0; i < it->second->getShape().getElementsCount(); ++i) {
                const float value = read(it->second, i);
                OPENVINO_ASSERT(
                    std::isfinite(value) && value == std::trunc(value) && value >= minimum && value <= maximum,
                    "Decode cache requires in-range integer zero points");
            }
        }
        auto rawDesc = std::make_shared<DnnlBlockedMemoryDesc>(ov::element::f16, Shape(dims));
        ov::float16 dummy(0.f);
        auto rawView = std::make_shared<Memory>(context->getEngine(), rawDesc, &dummy, false);
        MemoryArgs native = memory;
        native[ARG_WEI] = rawView;
        native.erase(ARG_WEI | ARG_ATTR_SCALES);
        native.erase(ARG_WEI | ARG_ATTR_ZERO_POINTS);
        MatMulAttrs attrs;
        attrs.transposeB = true;
        attrs.withBias = memory.at(ARG_BIAS)->getShape().getElementsCount() != 0;
        attrs.constantWeights = true;
        attrs.fcSemantic = true;
        shapeData = DnnlMatMulPrimitive::createShapeAgnosticData(attrs, native, context, false);
        primitive = DnnlMatMulPrimitive::create(native, attrs, context, shapeData);
        OPENVINO_ASSERT(primitive && primitive->weightsDesc()->getPrecision() == ov::element::f16,
                        "Cache would materialize non-FP16 weights");
        const size_t rawBytes = rawDesc->getCurrentMemSize();
        packedBytes = primitive->weightsDesc()->getCurrentMemSize();
        const size_t scratchBytes = primitive->scratchPadDesc()->getCurrentMemSize();
        OPENVINO_ASSERT(packedBytes <= std::numeric_limits<size_t>::max() - rawBytes);
        OPENVINO_ASSERT(scratchBytes <= std::numeric_limits<size_t>::max() - rawBytes - packedBytes);
        const size_t required = rawBytes + packedBytes + scratchBytes;
        if (!budget->reserve(required)) {
            if (budget->trace)
                std::fprintf(stderr, "[FC-CACHE] REJECT budget required=%zu name=%s\n", required, name.c_str());
            primitive.reset();
            shapeData.reset();
            failed = true;
            return;
        }
        charged = required;  // Reserve the peak before allocating raw + packed + scratch.
        auto raw = std::make_shared<Memory>(context->getEngine(), rawDesc);
        auto* data = static_cast<ov::float16*>(raw->getData());
        ov::parallel_for(n, [&](size_t nc) {
            for (size_t kc = 0; kc < k; ++kc) {
                const float zero = zp ? zp->at(nc, kc, n, k) : 0.f;
                const float scale = scales.at(nc, kc, n, k);
                OPENVINO_ASSERT(std::isfinite(scale));
                const ov::float16 sub(read(quant, nc * k + kc) - zero);
                data[nc * k + kc] = ov::float16(float(sub) * scale);
            }
        });
        weights = std::make_shared<Memory>(context->getEngine(), primitive->weightsDesc());
        const auto transposed =
            DnnlMatMulPrimitive::makeTransposedWeightDescriptor(rawDesc, primitive->weightsDesc(), attrs);
        Memory transposedRaw(context->getEngine(), transposed, raw->getData(), false);
        node::Reorder::reorderData(transposedRaw, *weights, context->getRuntimeCache(), context->getThreadPool());
        scratch = std::make_shared<Memory>(context->getEngine(), primitive->scratchPadDesc());
        args = shapeData->m_primAttrs.dnnlArgs;
        args[DNNL_ARG_WEIGHTS] = weights->getPrimitive();
        args[DNNL_ARG_SCRATCHPAD] = scratch->getPrimitive();
        args[DNNL_ARG_SRC] = dnnl::memory(primitive->srcDesc()->getDnnlDesc(), context->getEngine(), DNNL_MEMORY_NONE);
        args[DNNL_ARG_DST] = dnnl::memory(primitive->dstDesc()->getDnnlDesc(), context->getEngine(), DNNL_MEMORY_NONE);
        if (attrs.withBias)
            args[DNNL_ARG_BIAS] = memory.at(ARG_BIAS)->getPrimitive();
        srcDesc = memory.at(ARG_SRC)->getDescPtr();
        dstDesc = memory.at(ARG_DST)->getDescPtr();
        raw.reset();
        budget->release(rawBytes);
        charged -= rawBytes;
        if (budget->trace)
            std::fprintf(stderr,
                         "[FC-CACHE] CREATE raw=%zu packed=%zu scratch=%zu retained=%zu n=%zu k=%zu impl=%d name=%s\n",
                         rawBytes,
                         packedBytes,
                         scratchBytes,
                         charged,
                         n,
                         k,
                         int(primitive->implType()),
                         name.c_str());
    }

public:
    void prepare(const MemoryArgs& memory) {
        if (budget->limit == 0 || failed || (!budget->selected.empty() && !budget->selected.count(name)))
            return;
        try {
            const auto& dims = memory.at(ARG_WEI)->getShape().getStaticDims();
            OPENVINO_ASSERT(dims.size() == 2);
            auto sourceDims = VectorDims(memory.at(ARG_SRC)->getShape().getRank(), 1);
            auto outputDims = VectorDims(memory.at(ARG_DST)->getShape().getRank(), 1);
            OPENVINO_ASSERT(sourceDims.size() >= 2 && outputDims.size() >= 2);
            sourceDims.back() = dims[1];
            outputDims.back() = dims[0];
            alignas(64) float dummy[16] = {};
            MemoryArgs native = memory;
            native[ARG_SRC] = std::make_shared<Memory>(
                context->getEngine(),
                std::make_shared<DnnlBlockedMemoryDesc>(memory.at(ARG_SRC)->getPrecision(), Shape(sourceDims)),
                dummy,
                false);
            native[ARG_DST] = std::make_shared<Memory>(
                context->getEngine(),
                std::make_shared<DnnlBlockedMemoryDesc>(memory.at(ARG_DST)->getPrecision(), Shape(outputDims)),
                dummy,
                false);
            initialize(native);
        } catch (const std::exception& e) {
            reset_cache();
            failed = true;
            if (budget->trace)
                std::fprintf(stderr, "[FC-CACHE] REJECT preparation name=%s reason=%s\n", name.c_str(), e.what());
        }
    }
    DecodeCache(ExecutorPtr original,
                FCAttrs attrs,
                ExecutorContext::CPtr context,
                std::shared_ptr<Budget> budget,
                std::string name)
        : original(std::move(original)),
          fcAttrs(std::move(attrs)),
          context(std::move(context)),
          budget(std::move(budget)),
          name(std::move(name)) {
        if (this->budget->trace)
            std::fprintf(stderr,
                         "[FC-CACHE] WRAP name=%s selected=%d\n",
                         this->name.c_str(),
                         this->budget->selected.empty() || this->budget->selected.count(this->name));
    }
    bool update(const MemoryArgs& memory) override {
        active = false;
        const auto& dims = memory.at(ARG_SRC)->getShape().getStaticDims();
        size_t m = 1;
        for (size_t i = 0; i + 1 < dims.size(); ++i)
            m *= dims[i];
        if (budget->trace && !tracedUpdate && m == 1) {
            std::fprintf(stderr, "[FC-CACHE] DECODE rank=%zu k=%zu name=%s\n", dims.size(), dims.back(), name.c_str());
            tracedUpdate = true;
        }
        if (m == 1 && budget->limit != 0 && !failed && (budget->selected.empty() || budget->selected.count(name))) {
            try {
                if (!primitive)
                    initialize(memory);
                active = primitive && weights && srcDesc->isCompatible(memory.at(ARG_SRC)->getDesc()) &&
                         dstDesc->isCompatible(memory.at(ARG_DST)->getDesc());
            } catch (const std::exception& e) {
                reset_cache();
                failed = true;
                if (budget->trace)
                    std::fprintf(stderr, "[FC-CACHE] REJECT eligibility name=%s reason=%s\n", name.c_str(), e.what());
            }
        }
        return active || original->update(memory);
    }
    void execute(const MemoryArgs& memory) override {
        if (active) {
            if (budget->trace)
                ++decodeCalls;
            args[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
            args[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());
            primitive->execute(args);
        } else {
            if (budget->trace)
                ++compressedCalls;
            original->execute(memory);
        }
    }
    impl_desc_type implType() const override {
        return active ? primitive->implType() : original->implType();
    }
    void moveMemToNumaNode(int id) override {
        if (!active) {
            original->moveMemToNumaNode(id);
        } else if (cachedNuma != id) {
            (void)mbind_move(args.at(DNNL_ARG_WEIGHTS), id);
            (void)mbind_move(args.at(DNNL_ARG_SCRATCHPAD), id);
            cachedNuma = id;
        }
    }
    ~DecodeCache() override {
        if (charged && budget->trace)
            std::fprintf(stderr,
                         "[FC-CACHE] RELEASE decode=%zu compressed=%zu packed=%zu name=%s\n",
                         decodeCalls,
                         compressedCalls,
                         packedBytes,
                         name.c_str());
        reset_cache();
    }
};
}  // namespace fc_decode_cache
}  // namespace ov::intel_cpu
