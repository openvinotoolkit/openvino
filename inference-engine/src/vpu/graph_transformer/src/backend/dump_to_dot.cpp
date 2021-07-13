// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <climits>
#include <cstring>

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>

#include <precision_utils.h>
#include <legacy/graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/ie_helpers.hpp>

namespace vpu {

namespace {

std::string dataDotName(const Data& data) {
    std::ostringstream ostr;
    ostr << "data_" << static_cast<const void*>(data.get());
    return ostr.str();
}

std::string stageDotName(const Stage& stage) {
    std::ostringstream ostr;
    ostr << "stage_" << static_cast<const void*>(stage.get());
    return ostr.str();
}

void dumpStageToDot(DotSerializer& out, const Stage& stage, int stageExecIdx) {
    std::string stageColor = "gold";
    if (stageExecIdx < 0) {
        stageColor = "azure";
    }

    out.append("%s [", stageDotName(stage));
    {
        VPU_DOT_IDENT(out);

        out.append("shape=ellipse");
        out.append("style=filled");
        out.append("fillcolor=%s", stageColor);

        std::ostringstream caption;
        caption << "[" << stageExecIdx << " / " << stage->index() << "] Stage " << stage->name();

        DotLabel lbl(caption.str(), out);
        lbl.appendPair("type", stage->type());
        if (stage->origLayer() != nullptr) {
            lbl.appendPair("origLayer", stage->origLayer());
        }
        lbl.appendPair("numSHAVEs", stage->numSHAVEs());
        if (!stage->attrs().empty()) {
            lbl.appendPair("extraAttrs", stage->attrs());
        }
    }
    out.append("];");
}

}  // namespace

void BackEnd::dumpModelToDot(
        const Model& model,
        const std::string& fileName) {
    VPU_PROFILE(dumpModelToDot);

    std::ofstream file(fileName);
    if (!file.is_open()) {
        VPU_THROW_EXCEPTION << "Failed to open DOT file " << fileName;
    }

    DotSerializer out(file);

    out.append("digraph ie_vpu_model_view {");
    {
        VPU_DOT_IDENT(out);

        out.append("labelloc=top;");
        out.append("labeljust=left;");

        {
            DotLabel lbl("Graph " + model->name(), out);
            lbl.appendPair("batchSize", model->batchSize());
            if (!model->attrs().empty()) {
                lbl.appendPair("extraAttrs", model->attrs());
            }
        }

        //
        // Dump datas
        //

        for (const auto& data : model->datas()) {
            std::string dataColor = "white";
            if (data->usage() == DataUsage::Input) {
                dataColor = "green";
            } else if (data->usage() == DataUsage::Output) {
                dataColor = "deepskyblue";
            } else if (data->usage() == DataUsage::Const) {
                dataColor = "aquamarine";
            } else if (data->usage() == DataUsage::Temp) {
                dataColor = "cyan";
            } else if (data->usage() == DataUsage::Intermediate) {
                if (data->dataLocation().location == Location::BSS) {
                    dataColor = "cyan";
                } else if (data->dataLocation().location == Location::CMX) {
                    dataColor = "magenta";
                } else if (data->dataLocation().location == Location::Blob) {
                    dataColor = "aquamarine";
                } else if (data->dataLocation().location == Location::Input) {
                    dataColor = "green";
                } else if (data->dataLocation().location == Location::Output) {
                    dataColor = "deepskyblue";
                }
            }

            out.append("%s [", dataDotName(data));
            {
                VPU_DOT_IDENT(out);

                out.append("shape=box");
                out.append("style=filled");
                out.append("fillcolor=%s", dataColor);

                DotLabel lbl("Data " + data->name(), out);
                lbl.appendPair("usage", data->usage());
                lbl.appendPair("desc", data->desc());
                lbl.appendPair("requiredStrides", data->requiredStrides());
                lbl.appendPair("strides", data->strides());
                if (data->origData() != nullptr) {
                    lbl.appendPair("origData", data->origData());
                }
                if (data->content() != nullptr) {
                    if (data->desc().type() == DataType::U8) {
                        auto contentPtr = data->content()->get<uint8_t>();
                        auto count = data->desc().totalDimSize();

                        SmallVector<int, 8> temp(
                            contentPtr,
                            contentPtr + std::min(count, 8));

                        lbl.appendPair("content", temp);
                    } else if (data->desc().type() == DataType::FP16) {
                        auto contentPtr = data->content()->get<fp16_t>();
                        auto count = data->desc().totalDimSize();

                        SmallVector<float, 8> temp(std::min(count, 8));
                        ie::PrecisionUtils::f16tof32Arrays(temp.data(), contentPtr, temp.size());

                        lbl.appendPair("content", temp);
                    }
                }
                lbl.appendPair("memReqs", data->memReqs());
                lbl.appendPair("dataLocation", data->dataLocation().location);
                lbl.appendPair("dataOffset", data->dataLocation().offset);
                lbl.appendPair("dimsLocation", data->shapeLocation().dimsLocation);
                lbl.appendPair("dimsOffset", data->shapeLocation().dimsOffset);
                lbl.appendPair("stridesLocation", data->shapeLocation().stridesLocation);
                lbl.appendPair("stridesOffset", data->shapeLocation().stridesOffset);
                if (!data->attrs().empty()) {
                    lbl.appendPair("extraAttrs", data->attrs());
                }
            }
            out.append("];");
        }

        //
        // Dump stages
        //

        int stageExecIdx = 0;
        for (const auto& stage : model->getStages()) {
            if (stage->category() == StageCategory::Special) {
                dumpStageToDot(out, stage, -1);
            } else {
                dumpStageToDot(out, stage, stageExecIdx);
            }

            if (const auto injectedStage = stage->injectedStage()) {
                dumpStageToDot(out, injectedStage, stageExecIdx);
            }

            if (stage->category() != StageCategory::Special) {
                ++stageExecIdx;
            }
        }

        //
        // Dump Stage <-> Data edges
        //

        for (const auto& stage : model->getStages()) {
            for (const auto& inEdge : stage->inputEdges()) {
                out.append("%s -> %s [", dataDotName(inEdge->input()), stageDotName(stage));
                {
                    VPU_DOT_IDENT(out);

                    DotLabel lbl("StageInput", out);
                    lbl.appendPair("portInd", inEdge->portInd());
                    if (!inEdge->attrs().empty()) {
                        lbl.appendPair("extraAttrs", inEdge->attrs());
                    }
                }
                out.append("];");
            }

            for (const auto& outEdge : stage->outputEdges()) {
                out.append("%s -> %s [", stageDotName(stage), dataDotName(outEdge->output()));
                {
                    VPU_DOT_IDENT(out);

                    DotLabel lbl("StageOutput", out);
                    lbl.appendPair("portInd", outEdge->portInd());
                    if (!outEdge->attrs().empty()) {
                        lbl.appendPair("extraAttrs", outEdge->attrs());
                    }
                }
                out.append("];");
            }

            for (const auto& tempBufferEdge : stage->tempBufferEdges()) {
                out.append("%s -> %s [", dataDotName(tempBufferEdge->tempBuffer()), stageDotName(stage));
                {
                    VPU_DOT_IDENT(out);

                    DotLabel lbl("Temp buffer", out);
                    lbl.appendPair("portInd", tempBufferEdge->portInd());
                    if (!tempBufferEdge->attrs().empty()) {
                        lbl.appendPair("extraAttrs", tempBufferEdge->attrs());
                    }
                }
                out.append("];");
            }

            if (const auto injectedStage = stage->injectedStage()) {
                for (const auto& inEdge : injectedStage->inputEdges()) {
                    out.append("%s -> %s [", dataDotName(inEdge->input()), stageDotName(injectedStage));
                    {
                        VPU_DOT_IDENT(out);

                        out.append("style=dotted");

                        DotLabel lbl("StageInput", out);
                        lbl.appendPair("portInd", inEdge->portInd());
                        if (!inEdge->attrs().empty()) {
                            lbl.appendPair("extraAttrs", inEdge->attrs());
                        }
                    }
                    out.append("];");
                }

                for (const auto& outEdge : injectedStage->outputEdges()) {
                    out.append("%s -> %s [", stageDotName(injectedStage), dataDotName(outEdge->output()));
                    {
                        VPU_DOT_IDENT(out);

                        out.append("style=dotted");

                        DotLabel lbl("StageOutput", out);
                        lbl.appendPair("portInd", outEdge->portInd());
                        if (!outEdge->attrs().empty()) {
                            lbl.appendPair("extraAttrs", outEdge->attrs());
                        }
                    }
                    out.append("];");
                }

                for (const auto& tempBufferEdge : injectedStage->tempBufferEdges()) {
                    out.append("%s -> %s [", dataDotName(tempBufferEdge->tempBuffer()), stageDotName(injectedStage));
                    {
                        VPU_DOT_IDENT(out);

                        out.append("style=dotted");

                        DotLabel lbl("Temp buffer", out);
                        lbl.appendPair("portInd", tempBufferEdge->portInd());
                        if (!tempBufferEdge->attrs().empty()) {
                            lbl.appendPair("extraAttrs", tempBufferEdge->attrs());
                        }
                    }
                    out.append("];");
                }
            }
        }

        //
        // Dump Data<->Data edges
        //

        for (const auto& data : model->datas()) {
            if (auto edge = data->parentDataToDataEdge()) {
                out.append("%s -> %s [", dataDotName(edge->child()), dataDotName(edge->parent()));
                {
                    VPU_DOT_IDENT(out);

                    out.append("style=dotted");

                    DotLabel lbl("DataToDataAllocation", out);
                    lbl.appendPair("mode", edge->mode());
                    lbl.appendPair("order", edge->order());
                    if (!edge->attrs().empty()) {
                        lbl.appendPair("extraAttrs", edge->attrs());
                    }
                }
                out.append("];");
            }
        }

        //
        // Dump Data<->Data shape edges
        //

        for (const auto& data : model->datas()) {
            if (auto edge = data->parentDataToShapeEdge()) {
                out.append("%s -> %s [", dataDotName(edge->parent()), dataDotName(edge->child()));
                {
                    VPU_DOT_IDENT(out);

                    out.append("style=dotted");

                    DotLabel lbl("DataToShapeAllocation", out);
                }
                out.append("];");
            }
        }

        //
        // Dump Stage<->Stage edges
        //

        for (const auto& stage : model->getStages()) {
            if (const auto injectionEdge = stage->injectedStageEdge()) {
                out.append("%s -> %s [", stageDotName(stage), stageDotName(injectionEdge->child()));
                {
                    VPU_DOT_IDENT(out);

                    out.append("style=dashed");

                    DotLabel lbl("Injected Stage", out);
                    if (!injectionEdge->attrs().empty()) {
                        lbl.appendPair("extraAttrs", injectionEdge->attrs());
                    }
                }
                out.append("];");

                out.append("{");
                {
                    VPU_DOT_IDENT(out);

                    out.append("rank=same;");
                    out.append("%s, %s", stageDotName(stage), stageDotName(injectionEdge->child()));
                }
                out.append("}");
            }

            for (const auto& stageDependencyEdge : stage->childDependencyEdges()) {
                out.append("%s -> %s [", stageDotName(stage), stageDotName(stageDependencyEdge->child()));
                {
                    VPU_DOT_IDENT(out);

                    DotLabel lbl("Extra dependency", out);
                }
                out.append("];");
            }
        }
    }
    out.append("}");
}

}  // namespace vpu
