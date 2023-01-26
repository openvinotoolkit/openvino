// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "extension_mngr.h"
#include "weights_cache.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 ExtensionManager::Ptr extensionManager,
                 WeightsSharing::Ptr w_cache,
                 std::shared_ptr<std::mutex> sharedMutex,
                 bool isGraphQuantized)
        : config(config),
          extensionManager(extensionManager),
          weightsCache(w_cache),
          sharedMutex(sharedMutex),
          isGraphQuantizedFlag(isGraphQuantized) {
        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
        rtScratchPad = std::make_shared<DnnlScratchPad>(eng);
    }

    const Config& getConfig() const {
        return config;
    }

    ExtensionManager::Ptr getExtensionManager() const {
        return extensionManager;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }

    std::shared_ptr<std::mutex> getSharedMutex() const {
        return sharedMutex;
    }

    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return rtScratchPad;
    }

    dnnl::engine getEngine() const {
        return eng;
    }

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

    dnnl::reorder getReorderPrim(const dnnl::memory::desc& src, const dnnl::memory::desc& dest) const;
    void reorderData(const Memory &input, const Memory &output) const;

    template<typename K>
    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getPrim(const K & key) const;

    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getConvPrim(const dnnl::memory::desc& src,
                                                                      const dnnl::memory::desc& weight,
                                                                      const dnnl::memory::desc& bias,
                                                                      const dnnl::memory::desc& dst,
                                                                      const std::vector<ptrdiff_t>& stride,
                                                                      const std::vector<ptrdiff_t>& dilation,
                                                                      const std::vector<ptrdiff_t>& paddingL,
                                                                      const std::vector<ptrdiff_t>& paddingR,
                                                                      const dnnl::primitive_attr& attr,
                                                                      const impl_desc_type& implType) const;

    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getConvBackPrim(const dnnl::memory::desc& src,
                                                                          const dnnl::memory::desc& weight,
                                                                          const dnnl::memory::desc& dst,
                                                                          const std::vector<ptrdiff_t>& stride,
                                                                          const std::vector<ptrdiff_t>& dilation,
                                                                          const std::vector<ptrdiff_t>& paddingL,
                                                                          const std::vector<ptrdiff_t>& paddingR,
                                                                          const dnnl::primitive_attr& attr,
                                                                          const impl_desc_type& implType) const;

    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getDeconvPrim(const dnnl::memory::desc& src,
                                                                        const dnnl::memory::desc& weight,
                                                                        const dnnl::memory::desc& bias,
                                                                        const dnnl::memory::desc& dst,
                                                                        const std::vector<ptrdiff_t>& stride,
                                                                        const std::vector<ptrdiff_t>& dilation,
                                                                        const std::vector<ptrdiff_t>& paddingL,
                                                                        const std::vector<ptrdiff_t>& paddingR,
                                                                        const dnnl::primitive_attr& attr,
                                                                        const impl_desc_type& implType) const;

    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getInnerProductPrim(const dnnl::memory::desc& src,
                                                                              const dnnl::memory::desc& weight,
                                                                              const dnnl::memory::desc& bias,
                                                                              const dnnl::memory::desc& dst,
                                                                              const dnnl::primitive_attr& attr,
                                                                              const impl_desc_type& implType) const;

    std::pair<dnnl::primitive, dnnl::primitive_desc_base> getMatMulPrim(const dnnl::memory::desc& src,
                                                                        const dnnl::memory::desc& weight,
                                                                        const dnnl::memory::desc& bias,
                                                                        const dnnl::memory::desc& dst,
                                                                        const dnnl::primitive_attr& attr,
                                                                        const impl_desc_type& implType) const;

private:
    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data
    std::shared_ptr<std::mutex> sharedMutex;  // mutex for protection of type-relaxed Op in clone_model()

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;
    static dnnl::engine eng;  // onednn engine (singleton)
};

}  // namespace intel_cpu
}  // namespace ov
