// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class DetectionOutput : public Node {
public:
    DetectionOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

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
    float objScore = 0.0f;

    float confidenceThreshold = 0.0f;
    float sparsityThreshold = 0.03f;
    int topK = 0;
    float NMSThreshold = 0.0f;
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

    enum CodeType {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

    int confInfoLen = 0;
    bool isSparsityWorthwhile = false;

    inline void getActualPriorNum(const float* priorData, int* numPriorsActual, int n);

    inline void confReorderDense(const float* confData, const float* ARMConfData, float* reorderedConfData);

    inline void confFilterCF(const float* pConf, int* pindices, int* pbuffer, int* detectionsData, const int& n);

    inline void confFilterMX(const float* confData,
                             const float* ARMConfData,
                             float* reorderedConfData,
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

    inline void decodeBBoxes(const float* prior_data,
                             const float* loc_data,
                             const float* variance_data,
                             float* decoded_bboxes,
                             float* decoded_bbox_sizes,
                             int* num_priors_actual,
                             int n,
                             const int& offs,
                             const int& pr_size,
                             bool decodeType = true,
                             const int* conf_info_h = nullptr,
                             const int* conf_info_v = nullptr);  // decodeType is false after ARM

    inline void NMSCF(int* indicesIn, int& detections, int* indicesOut, const float* bboxes, const float* boxSizes);

    inline void NMSMX(int* indicesIn, int* detections, int* indicesOut, const float* bboxes, const float* sizes);

    inline void topk(const int* indicesIn, int* indicesOut, const float* conf, int n, int k);

    inline void generateOutput(float* reorderedConfData,
                               int* indicesData,
                               int* detectionsData,
                               float* decodedBboxesData,
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
