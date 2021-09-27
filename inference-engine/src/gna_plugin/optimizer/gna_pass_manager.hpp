// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <ie_common.h>

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
    virtual const bool& isLowPrecision() const = 0;
    virtual InferenceEngine::CNNNetwork &getNetwork() = 0;
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
 * brief split ofver channels for Elementwise-layer to avoid GNA-HW limitation of 65 elements per eltwise
 */
DECL_PASS(EltwiseSplitOverChannels);
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
 * @brief GNA doesn't provide intermediate results (sums) when the layer is fused with activation.
 * When more layers use the sums as inputs (beside the activation) then the diagonal layer
 * is inserted before the activation to forbid the fusing and make the sums exposed.
 * This is observed in the multiple_activations_onGNA_INT16 test.
 */
DECL_PASS(ForbidActivationFusing);

/**
 * @brief copy layer insertion required in cases where input layer does not have output memory
 */
DECL_PASS(InsertCopyLayer);

/**
 * @brief aligning filter layer insertion required in cases when split/slice have output connections on not aligned addresses
 */
DECL_PASS(InsertSplitAligningFilter);

/**
* @brief Pass that flattens trivial concatenations inputs and output and changes its axis to 1
*/
DECL_PASS(FlattenTrivialConcat);

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
* @brief in cases that network output layer is connected to only one layer which is activation additional identity is inserted
* so the operation is not fused with the activation allowing to get te results from said layer
*/
DECL_PASS(BreakFusingOfOutputLayers);

/**
 * @brief insert identity at the output of LSTMCell which fixes cases where data is not propagated correctly through network
 * and LSTMCell returns all zeroes
 */
DECL_PASS_BEFORE_COPY(InsertIdentityToLSTMCell);

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
 * @brief remove concat layers with single input
*/
DECL_PASS_BEFORE_COPY(RemoveSingleInputConcat);

/**
 * @brief removed extra identity layer for multi-output
 */
DECL_PASS(FuseMultipleIdentities);

/**
* @brief Brodcast data in Const layer
*/
DECL_PASS(BroadcastConst);

/**
* @brief runs static quantisation on given floating weights and replaces fakeQuantize with constblobs
*/
DECL_PASS(FuseFQIntoWeights);

/**
* @brief remove all fake quantize layers while moving it's settings into QuantParams for certain layer
*/
DECL_PASS(MoveFakeQuantizeLayerIntoQuantParams);

/**
* @brief convert FullyConnected, ScaleShift and Eltwise layers weights order from NCHW to NHWC.
* Information for transposition is found from convolution/pooling input or output dimensions.
* Convolution weights are transposed in finalizeConvolution1DPrimitive() method (gna_graph_compiler.cpp).
* They are transposed for the both, NCHW and NHWC models since MO always stores them in NCHW layout.
*/
DECL_PASS(TransposeWeightsFromNCHWToNHWC);

struct PassManagerSettings {
    /// @brief whether to run passes before copy
    bool runBeforeCopy;
    bool lowPrecision;
};


class PassManager : public IPassManager, public std::enable_shared_from_this<PassManager> {
    PassManagerSettings settings;
    InferenceEngine::CNNNetwork network;
    std::vector<std::shared_ptr<Pass>> passes;
    std::map<std::string, int> intMap;

public:
    explicit PassManager(PassManagerSettings settings, InferenceEngine::CNNNetwork network) noexcept
    : settings(settings)
    , network(network) {}

    template <class T>
    void registerPass() {
        passes.push_back(std::make_shared<T>(shared_from_this()));
    }
    int & getIntVar(std::string name) override {
        return intMap[name];
    }
    const bool& isLowPrecision() const override {
        return settings.lowPrecision;
    }
    InferenceEngine::CNNNetwork& getNetwork() override {
        return network;
    }
    /**
     * @brief returns number of passes have been passed
     * @param index - start index start index of first pass - used only in logging right now
     */
    int run(int index = 0);
};

}  // namespace GNAPluginNS
