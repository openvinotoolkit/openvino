// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <extension/ext_base.hpp>
#include <extension/ext_base.cpp>
#include <extension/ext_list.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class FakeLayerPLNImpl: public ExtLayerBase {
public:
    explicit FakeLayerPLNImpl(const CNNLayer* layer) {
        try {
            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

class FakeLayerBLKImpl: public ExtLayerBase {
public:
    explicit FakeLayerBLKImpl(const CNNLayer* layer) {
        try {
#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif
            addConfig(layer, {{blk_layout, false, 0}}, {{blk_layout, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

REG_FACTORY_FOR(ImplFactory<FakeLayerPLNImpl>, FakeLayerPLN);
REG_FACTORY_FOR(ImplFactory<FakeLayerBLKImpl>, FakeLayerBLK);

}
}
}
