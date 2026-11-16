// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// An ONNX GraphIterator built against the FROZEN, earlier version of the public header (graph_iterator_v1.hpp).
// It stands for a prebuilt third-party iterator library: its class has no GraphIteratorV2 base, its vtable has no slots
// for anything newer than the original virtual functions, and its object has no storage for anything newer.
// The ONNX frontend tests load it through the plain C functions below and drive it with the current frontend.

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include "graph_iterator_v1.hpp"

#if defined(_WIN32)
#    define LEGACY_ITERATOR_API extern "C" __declspec(dllexport)
#else
#    define LEGACY_ITERATOR_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

std::atomic<size_t> g_get_model_dir_calls{0};
std::atomic<size_t> g_get_metadata_calls{0};

class LegacyIterator : public ov::frontend::onnx::GraphIterator {
public:
    explicit LegacyIterator(std::filesystem::path model_dir) : m_model_dir(std::move(model_dir)) {}

    size_t size() const override {
        return 0;
    }
    void reset() override {}
    void next() override {}
    bool is_end() const override {
        return true;
    }
    std::shared_ptr<ov::frontend::onnx::DecoderBase> get_decoder() const override {
        return nullptr;
    }
    int64_t get_opset_version(const std::string&) const override {
        return 1;
    }

    // Returned by value, as the earlier interface required.
    std::map<std::string, std::string> get_metadata() const override {
        ++g_get_metadata_calls;
        return {{"legacy_key", "legacy_value"}};
    }
    std::filesystem::path get_model_dir() const override {
        ++g_get_model_dir_calls;
        return m_model_dir;
    }

private:
    std::filesystem::path m_model_dir;
};

}  // namespace

// Declarations of the exported functions. They precede the definitions so that -Wmissing-declarations (enabled for
// the whole source tree) stays quiet. The tests declare the same functions on their side.
LEGACY_ITERATOR_API void* legacy_iterator_create(const char* model_dir);
LEGACY_ITERATOR_API void legacy_iterator_destroy(void* iterator);
LEGACY_ITERATOR_API size_t legacy_iterator_get_model_dir_calls();
LEGACY_ITERATOR_API size_t legacy_iterator_get_metadata_calls();
LEGACY_ITERATOR_API size_t legacy_iterator_base_class_size();

/// Creates the iterator. The caller owns it and must release it with legacy_iterator_destroy().
/// The pointer is the address of the ov::frontend::onnx::GraphIterator base subobject.
LEGACY_ITERATOR_API void* legacy_iterator_create(const char* model_dir) {
    g_get_model_dir_calls = 0;
    g_get_metadata_calls = 0;
    std::unique_ptr<ov::frontend::onnx::GraphIterator> iterator =
        std::make_unique<LegacyIterator>(std::filesystem::path(model_dir ? model_dir : ""));
    return iterator.release();
}

LEGACY_ITERATOR_API void legacy_iterator_destroy(void* iterator) {
    delete static_cast<ov::frontend::onnx::GraphIterator*>(iterator);
}

LEGACY_ITERATOR_API size_t legacy_iterator_get_model_dir_calls() {
    return g_get_model_dir_calls;
}

LEGACY_ITERATOR_API size_t legacy_iterator_get_metadata_calls() {
    return g_get_metadata_calls;
}

/// sizeof() of the base class as this library saw it. The current header must agree.
LEGACY_ITERATOR_API size_t legacy_iterator_base_class_size() {
    return sizeof(ov::frontend::onnx::GraphIterator);
}
