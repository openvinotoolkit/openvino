// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <ie_common.h>
#include "gna_plugin_policy.hpp"

namespace GNAPluginNS {
/**
 * @brief interface for gna-pass, special transformer that will be run on input network in order to generate GNABlob
 */
class Pass {
 public:
    virtual ~Pass() = default;
    virtual void attach(const std::vector<InferenceEngine::CNNLayerPtr> & layers)  = 0;
    virtual std::string getName() const = 0;
    virtual void run() = 0;
    virtual bool runBeforeCopyPass() { return false; }
};
/**
 * Passmanager interface available for individual passes, usually needed to store shared data between passes
 */
class IPassManager {
public:
    virtual ~IPassManager() = default;
    virtual int &getIntVar(std::string name) = 0;
    virtual const Policy &getPolicy() const = 0;
    virtual const InferenceEngine::CNNNetPtr &getNetwork() const = 0;
};

class BasePass : public Pass {
 protected:
    const std::vector<InferenceEngine::CNNLayerPtr> * pLayers = nullptr;
    std::weak_ptr<IPassManager> mgr;
 public:
    BasePass() = default;
    explicit BasePass(std::shared_ptr<IPassManager> mgr) : mgr(mgr) {}
    void attach(const std::vector<InferenceEngine::CNNLayerPtr> & layersToAttach) override {
        pLayers = &layersToAttach;
    }
 protected:
    std::shared_ptr<IPassManager> getPassManager();
};

#define DECL_PASS(PassName) \
class PassName##Pass : public BasePass {\
 public:\
    using BasePass::BasePass;\
    void run() override;\
    std::string getName() const override { return #PassName;}\
}

#define DECL_PASS_BEFORE_COPY(PassName) \
class PassName##Pass : public BasePass {\
 public:\
    using BasePass::BasePass;\
    void run() override;\
    bool runBeforeCopyPass() override { return true; };\
    std::string getName() const override { return #PassName;}\
}

/**
* @brief GNA affine layers are always have activation attached, while IR not
*/
DECL_PASS(InsertIdentityLayer);

/**
 * @brief GNA cannot support broadcast - so we will tile weights and biases for scaleshift layer
 */
DECL_PASS(SubstituteScaleShiftBroadCast);

/**
 * @brief GNA convolution layers have deinterleaved layout, while affine one doesn't
 * so between convolution and affine layers permute layers need to be inserted,
 * current MO approach is to insert such permutations
 * since GNA-HW already support conv->affine in permuted for, this pass inverses MO behavior
 * so its remove permutations of certain form conv->conv, and between conv->affine
 * and insert permutation between conv->affine if they are missed in IR
 * @param layers
 */
DECL_PASS(ReversePermutations);

/**
 * @brief Pass support --disable_nhwc_to_nchw option in MO
 * @param layers
 */
DECL_PASS(RemovePermutationsNHWCToNCHW);

/**
 * brief @search for specific patter in the graph (6 layers are replaced by single one)
 */
DECL_PASS(SubstitutePRelu);

/**
 * brief @search for specific patter in the graph (5 layers are replaced by single one)
 */
DECL_PASS(SubstituteSoftSign);

/**
 * diagonal layer insertion required in cases where activation followed by split layers, or any other
 * topology changing layers
 */
DECL_PASS(InsertDiagonalLayer);

/**
 * @brief MaxPool can be reordered with activation, on GNA there is a strategy to have conv->maxpool->activation
 * it means maxpool receives 4 bytes, and produces 4 bytes
 */
DECL_PASS(ReorderMaxPool);
/**
 * @brief GNA doesn't support multiple activations fused with functional layer
 * currently for n activations for the layer X, it will be 1 PWL identity inserted, and n diagonal layers.
 * if one of activations is already identity, n-1 diagonal layers will be inserted
 */
DECL_PASS(HandleMultipleActivationsForTheLayer);

/**
 * @brief copy layer insertion required in cases where input layer does not have output memory
 */
DECL_PASS(InsertCopyLayer);

/**
 * @brief aligning filter layer insertion required in cases when split/slice have output connections on not aligned addresses
 */
DECL_PASS(InsertSplitAligningFilter);

/**
 * @brief concat-aligning filter layer insertion required in cases when concat inputs size are not 64-aligned
 */
DECL_PASS(InsertConcatAligningFilter);

/**
 * @brief concat-aligning filter if inserted need to be folowed by left aligning inupt in multiple inputs to concate case
 * or just followed by first input to concate. This cannot be done in inserting concat aliging phase
 */
DECL_PASS(ReorderConcatInputs);

/**
* @brief unrolled LSTM cell layer in supported GNA primitives
*/
DECL_PASS_BEFORE_COPY(UnrollLSTMCell);

/**
* @brief unrolled Tensor Iterator layer in supported GNA layers
*/
DECL_PASS_BEFORE_COPY(UnrollTI);

/**
* @brief removed const layer before reshape layer
*/
DECL_PASS_BEFORE_COPY(RemoveConst);

/**
 * @brief removed extra identity layer for multi-output
 */
DECL_PASS(FuseMultipleIdentities);

/**
* @brief Brodcast data in Const layer
*/
DECL_PASS(BroadcastConst);

struct PassManagerSettings {
    Policy policy;
    /// @brief whether to run passes before copy
    bool runBeforeCopy;
};


class PassManager : public IPassManager, public std::enable_shared_from_this<PassManager> {
    PassManagerSettings settings;
    InferenceEngine::CNNNetPtr network;
    std::vector<std::shared_ptr<Pass>> passes;
    std::map<std::string, int> intMap;

public:
    explicit PassManager(PassManagerSettings settings, InferenceEngine::CNNNetPtr network) noexcept
    : settings(settings)
    , network(network) {}

    template <class T>
    void registerPass() {
        passes.push_back(std::make_shared<T>(shared_from_this()));
    }
    int & getIntVar(std::string name) override {
        return intMap[name];
    }
    const Policy & getPolicy() const override {
        return settings.policy;
    }
    const InferenceEngine::CNNNetPtr & getNetwork() const override {
        return network;
    }
    /**
     * @brief returns number of passes have been passed
     * @param index - start index start index of first pass - used only in logging right now
     */
    int run(int index = 0);
};

}  // namespace GNAPluginNS
