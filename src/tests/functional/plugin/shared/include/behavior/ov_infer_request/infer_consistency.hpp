// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "openvino/openvino.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace behavior {
// for deviceConfigs, the deviceConfigs[0] is target device which need to be tested.
// deviceConfigs[1], deviceConfigs[2],deviceConfigs[n] are the devices which will
// be compared with target device, the result of target should be in one of the compared
// device.
using OVInferConsistencyTestParamsTuple = typename std::tuple <
    unsigned int, //inferRequst nums per model
    unsigned int, //infer nums wil do per  inferRequest
    std::vector<std::pair<std::string, ov::AnyMap>> // devicesConfigs
    >;
struct InferContext {
    ov::InferRequest _inferRequest;
    std::vector<ov::Tensor> _outputs;
    std::vector<ov::Tensor> _inputs;
};

struct ModelContext {
    ov::CompiledModel _model;
    std::vector<InferContext> _inferContexts;
};

class OVInferConsistencyTest : public
    testing::WithParamInterface<OVInferConsistencyTestParamsTuple>,
    public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>&
        obj);

protected:
    void SetUp() override;
    void TearDown() override;
    void InferCheck(bool isSync);
    // with different index, will fill different input, the index should > 0.
    void FillInput(InferContext& inferContext, int index);
    std::vector<ov::Tensor> GetAllOutputs(ov::CompiledModel& model,
        ov::InferRequest& infer);
    bool IsEqual(std::vector<ov::Tensor>& a,
        std::vector<ov::Tensor>& b);
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::vector<std::pair<std::string, ov::AnyMap>> _deviceConfigs;
    std::vector<ModelContext> _modelContexts;
    unsigned int _inferReqNumPerModel;
    unsigned int _inferNumPerInfer;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
