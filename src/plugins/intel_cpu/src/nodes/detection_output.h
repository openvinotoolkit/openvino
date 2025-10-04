// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class DetectionOutput : public Node {
public:
    DetectionOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    static const int ID_LOC = 0;
    static const int ID_CONF = 1;
    static const int ID_PRIOR = 2;
    static const int ID_ARM_CONF = 3;
    static const int ID_ARM_LOC = 4;

    int imgNum = 0;
    int priorsNum = 0;
    int classesNum = 0;
    int priorSize = 4;
    bool isPriorsPerImg = false;
    bool isShareLoc = false;
    int locNumForClasses = 0;
    bool withAddBoxPred = false;
    float objScore = 0.0F;

    float confidenceThreshold = 0.0F;
    float sparsityThreshold = 0.03F;
    int topK = 0;
    float NMSThreshold = 0.0F;
    bool clipBeforeNMS = false;
    bool clipAfterNMS = false;
    int backgroundClassId = 0;
    bool decreaseClassId = false;
    int keepTopK = 0;

    bool varianceEncodedInTarget = false;
    bool normalized = false;
    int codeType = 1;
    int imgWidth = 0;
    int imgHeight = 0;
    int coordOffset = 0;
    int cacheSizeL3 = 0;

    enum CodeType : uint8_t {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

    int confInfoLen = 0;
    bool isSparsityWorthwhile = false;

    inline void getActualPriorNum(const float* priorData, int* numPriorsActual, int n) const;

    inline void confReorderDense(const float* confData, const float* ARMConfData, float* reorderedConfData) const;

    inline void confFilterCF(const float* pConf, int* pindices, int* pbuffer, int* detectionsData, const int& n);

    inline void confFilterMX(const float* confData,
                             const float* ARMConfData,
                             const float* reorderedConfData,
                             int* indicesData,
                             int* indicesBufData,
                             int* detectionsData,
                             const int& n);

    inline void confReorderAndFilterSparsityCF(const float* confData,
                                               const float* ARMConfData,
                                               float* reorderedConfData,
                                               int* indicesData,
                                               int* indicesBufData,
                                               int* detectionsData);

    inline void confReorderAndFilterSparsityMX(const float* confData,
                                               const float* ARMConfData,
                                               float* reorderedConfData,
                                               int* indicesData,
                                               int* indicesBufData,
                                               int* detectionsData);

    inline void decodeBBoxes(const float* prior_coords,
                             const float* location_deltas,
                             const float* variances,
                             float* decodedBboxes,
                             float* decodedBboxSizes,
                             const int* numPriorsActual,
                             int n,
                             const int& offs,
                             const int& priorSize,
                             bool decodeType = true,
                             const int* confInfoH = nullptr,
                             const int* confInfoV = nullptr) const;  // decodeType is false after ARM

    inline void NMSCF(const int* indicesIn,
                      int& detections,
                      int* indicesOut,
                      const float* bboxes,
                      const float* boxSizes) const;

    inline void NMSMX(const int* indicesIn,
                      int* detections,
                      int* indicesOut,
                      const float* bboxes,
                      const float* sizes) const;

    static inline void topk(const int* indicesIn, int* indicesOut, const float* conf, int n, int k);

    inline void generateOutput(const float* reorderedConfData,
                               const int* indicesData,
                               const int* detectionsData,
                               const float* decodedBboxesData,
                               float* dstData);

    std::vector<float> decodedBboxes;
    std::vector<int> indicesBuffer;
    std::vector<int> indices;
    std::vector<int> detectionsCount;
    std::vector<float> reorderedConf;
    std::vector<float> bboxSizes;
    std::vector<int> numPriorsActual;
    std::vector<int> confInfoForPrior;
};

}  // namespace ov::intel_cpu::node
