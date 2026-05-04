// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <any>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
class Model;
class ICompiledModel;
}  // namespace ov

namespace ov::npuw {
class CompiledModel;
class IBaseInferRequest;
struct Function;
struct Subgraph;
namespace online {
class Snapshot;
}
}  // namespace ov::npuw

namespace ov::npuw::v1::subgraphs {

class Context {
public:
    template <typename T>
    T& put(T value) {
        auto [it, inserted] = m_entries.insert_or_assign(std::type_index(typeid(T)), std::any(std::move(value)));
        return *std::any_cast<T>(&it->second);
    }

    template <typename T, typename... Args>
    T& emplace(Args&&... args) {
        auto [it, inserted] = m_entries.insert_or_assign(std::type_index(typeid(T)),
                                                         std::any(std::in_place_type<T>, std::forward<Args>(args)...));
        return *std::any_cast<T>(&it->second);
    }

    template <typename T>
    bool contains() const {
        return m_entries.find(std::type_index(typeid(T))) != m_entries.end();
    }

    template <typename T>
    void erase() {
        m_entries.erase(std::type_index(typeid(T)));
    }

    template <typename T>
    T* get_if() {
        auto it = m_entries.find(std::type_index(typeid(T)));
        return it == m_entries.end() ? nullptr : std::any_cast<T>(&it->second);
    }

    template <typename T>
    const T* get_if() const {
        auto it = m_entries.find(std::type_index(typeid(T)));
        return it == m_entries.end() ? nullptr : std::any_cast<T>(&it->second);
    }

    template <typename T>
    T& get() {
        auto* value = get_if<T>();
        if (!value) {
            OPENVINO_THROW("NPUW subgraph pipeline context is missing entry for type ", typeid(T).name());
        }
        return *value;
    }

    template <typename T>
    const T& get() const {
        auto* value = get_if<T>();
        if (!value) {
            OPENVINO_THROW("NPUW subgraph pipeline context is missing entry for type ", typeid(T).name());
        }
        return *value;
    }

    bool empty() const {
        return m_entries.empty();
    }

    std::size_t size() const {
        return m_entries.size();
    }

private:
    std::unordered_map<std::type_index, std::any> m_entries;
};

struct Registration {
    std::string group;
    std::string name;
    std::vector<std::string> patterns;
    std::size_t order = 0u;

    bool empty() const {
        return group.empty() && name.empty() && patterns.empty() && order == 0u;
    }
};

struct CompiledPipeline;
struct InferContext;
struct PartitioningCallbacks {
    std::function<std::shared_ptr<ov::Model>(const std::string&)> find_tagged_model;
};

struct CompileContext {
    std::shared_ptr<ov::Model>& model;
    ov::SoPtr<ov::ICompiledModel>& compiled_model;
    const std::vector<std::string>& devices;
    std::function<ov::SoPtr<ov::ICompiledModel>(const std::shared_ptr<ov::Model>&,
                                                const std::string&,
                                                const std::vector<std::string>&)>
        compile_model;
};

struct FunctionPipeline {
    Registration registration;
    Context context;
    std::function<void(ov::npuw::Function&, Context&)> partition_stage;
    std::function<void(CompiledPipeline&, Context&)> compile_stage;
};

class ISubgraphBehavior {
public:
    using Ptr = std::unique_ptr<ISubgraphBehavior>;

    virtual void prologue(InferContext&) {}
    virtual void run(InferContext&) = 0;
    virtual void epilogue(InferContext&) {}

    virtual ~ISubgraphBehavior() = default;
};

using RuntimeBehaviorFactory = std::function<ISubgraphBehavior::Ptr(const Context&)>;
using CompileExecutor = std::function<void(CompileContext&)>;

struct RuntimeBehaviorSpec {
    Registration registration;
    Context context;
    RuntimeBehaviorFactory factory;
    bool handles_function_prologue = false;
};

struct CompiledPipeline {
    Registration registration;
    Context context;
    CompileExecutor compile_executor;
    std::optional<RuntimeBehaviorSpec> runtime_behavior;
    bool is_function_call = false;
    std::optional<std::size_t> function_body_subgraph_idx;
};

struct InferContext {
    ov::npuw::CompiledModel& compiled_model;
    ov::npuw::IBaseInferRequest& infer_request;
    std::size_t subgraph_idx = 0u;
    std::size_t real_subgraph_idx = 0u;
    std::function<void()> legacy_infer;
    std::function<void()> opaque_prologue;
    std::function<void()> opaque_run;
};

using PostLegacyHook = std::function<void(InferContext&)>;

class DirectBehavior final : public ISubgraphBehavior {
public:
    using Runner = std::function<void(InferContext&)>;

    explicit DirectBehavior(Runner runner = {}) : m_runner(std::move(runner)) {}

    void run(InferContext& ctx) override {
        if (m_runner) {
            m_runner(ctx);
        }
    }

private:
    Runner m_runner;
};

inline ISubgraphBehavior::Ptr make_direct_behavior() {
    return std::make_unique<DirectBehavior>();
}

struct PatternRegistration {
    using PartitionStage = std::function<void(ov::npuw::Function&, Context&)>;
    using CompileStage = std::function<void(CompiledPipeline&, Context&)>;
    using MatcherRegistrar = std::function<
        void(ov::pass::GraphRewrite&, const std::shared_ptr<ov::npuw::online::Snapshot>&, const std::string&)>;

    std::size_t id = 0u;
    std::string pattern;
    std::string tag;
    std::string group;
    std::string name;
    PartitionStage partition_stage;
    CompileStage compile_stage;
    RuntimeBehaviorFactory runtime_factory;
    MatcherRegistrar matcher_registrar;
};

class PatternRegistry;

class ScopedPatternRegistration {
public:
    ScopedPatternRegistration() = default;
    ScopedPatternRegistration(PatternRegistry* registry, std::size_t id) : m_registry(registry), m_id(id) {}
    ScopedPatternRegistration(const ScopedPatternRegistration&) = delete;
    ScopedPatternRegistration& operator=(const ScopedPatternRegistration&) = delete;
    ScopedPatternRegistration(ScopedPatternRegistration&& other) noexcept
        : m_registry(std::exchange(other.m_registry, nullptr)),
          m_id(std::exchange(other.m_id, 0u)) {}
    ScopedPatternRegistration& operator=(ScopedPatternRegistration&& other) noexcept {
        if (this != &other) {
            reset();
            m_registry = std::exchange(other.m_registry, nullptr);
            m_id = std::exchange(other.m_id, 0u);
        }
        return *this;
    }
    ~ScopedPatternRegistration() {
        reset();
    }

    void reset();

private:
    PatternRegistry* m_registry = nullptr;
    std::size_t m_id = 0u;
};

template <typename Pattern>
class OnPatternBuilder {
public:
    explicit OnPatternBuilder(PatternRegistry& registry) : m_registry(registry) {
        m_registration.pattern = Pattern::pattern_name();
        m_registration.tag = Pattern::isolation_tag();
        m_registration.group = Pattern::group_name();
        m_registration.name = Pattern::pattern_name();
        m_registration.matcher_registrar = [](ov::pass::GraphRewrite& rewr,
                                              const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                              const std::string& isol_tag) {
            rewr.add_matcher<Pattern>(snapshot, isol_tag);
        };
    }

    OnPatternBuilder& tag(std::string value) {
        m_registration.tag = std::move(value);
        return *this;
    }

    OnPatternBuilder& group(std::string value) {
        m_registration.group = std::move(value);
        return *this;
    }

    OnPatternBuilder& at_partition(PatternRegistration::PartitionStage stage) {
        m_registration.partition_stage = std::move(stage);
        return *this;
    }

    OnPatternBuilder& at_compile(PatternRegistration::CompileStage stage) {
        m_registration.compile_stage = std::move(stage);
        return *this;
    }

    OnPatternBuilder& at_runtime(RuntimeBehaviorFactory factory) {
        m_registration.runtime_factory = std::move(factory);
        return *this;
    }

    ScopedPatternRegistration scoped();

private:
    PatternRegistry& m_registry;
    PatternRegistration m_registration;
};

class PatternRegistry {
public:
    template <typename Pattern>
    OnPatternBuilder<Pattern> on() {
        return OnPatternBuilder<Pattern>(*this);
    }

    ScopedPatternRegistration add(PatternRegistration registration) {
        registration.id = ++m_next_id;
        m_registrations.push_back(std::move(registration));
        return ScopedPatternRegistration(this, m_registrations.back().id);
    }

    void erase(std::size_t id) {
        m_registrations.erase(std::remove_if(m_registrations.begin(),
                                             m_registrations.end(),
                                             [id](const PatternRegistration& registration) {
                                                 return registration.id == id;
                                             }),
                              m_registrations.end());
    }

    bool register_matcher(ov::pass::GraphRewrite& rewr,
                          const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                          const std::string& pattern,
                          const std::string& isol_tag) const {
        bool handled = false;
        for (const auto& registration : m_registrations) {
            if (registration.pattern != pattern || !registration.matcher_registrar) {
                continue;
            }
            registration.matcher_registrar(rewr, snapshot, isol_tag);
            handled = true;
        }
        return handled;
    }

    void apply(ov::npuw::Function& function) const;
    void apply(ov::npuw::Subgraph& subgraph) const;
    void append_from(const PatternRegistry& other);

private:
    std::vector<PatternRegistration> m_registrations;
    std::size_t m_next_id = 0u;
};

inline void ScopedPatternRegistration::reset() {
    if (m_registry != nullptr) {
        m_registry->erase(m_id);
        m_registry = nullptr;
        m_id = 0u;
    }
}

template <typename Pattern>
inline ScopedPatternRegistration OnPatternBuilder<Pattern>::scoped() {
    return m_registry.add(std::move(m_registration));
}

}  // namespace ov::npuw::v1::subgraphs
