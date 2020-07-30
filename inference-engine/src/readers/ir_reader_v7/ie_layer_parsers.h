// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <debug.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_format_parser.h"
#include "xml_parse_utils.h"

inline pugi::xml_node GetChild(const pugi::xml_node& node, std::vector<std::string> tags, bool failIfMissing = true) {
    for (auto tag : tags) {
        pugi::xml_node dn = node.child(tag.c_str());
        if (!dn.empty()) return dn;
    }
    if (failIfMissing)
        THROW_IE_EXCEPTION << "missing <" << InferenceEngine::details::dumpVec(tags)
                           << "> Tags at offset :" << node.offset_debug();
    return pugi::xml_node();
}

using namespace XMLParseUtils;

namespace InferenceEngine {
namespace details {
template <class LT>
class LayerCreator : public BaseCreator {
public:
    explicit LayerCreator(const std::string& type): BaseCreator(type) {}

    CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) override {
        auto res = std::make_shared<LT>(layerParsePrms.prms);

        if (res->type == "Quantize") res->type = "FakeQuantize";

        if (std::is_same<LT, FullyConnectedLayer>::value) {
            layerChild[res->name] = {"fc", "fc_data", "data"};
        } else if (std::is_same<LT, NormLayer>::value) {
            layerChild[res->name] = {"lrn", "norm", "norm_data", "data"};
        } else if (std::is_same<LT, CropLayer>::value) {
            layerChild[res->name] = {"crop", "crop-data", "data"};
        } else if (std::is_same<LT, BatchNormalizationLayer>::value) {
            layerChild[res->name] = {"batch_norm", "batch_norm_data", "data"};
        } else if ((std::is_same<LT, EltwiseLayer>::value)) {
            layerChild[res->name] = {"elementwise", "elementwise_data", "data"};
        } else {
            layerChild[res->name] = {"data", tolower(res->type) + "_data", tolower(res->type)};
        }

        pugi::xml_node dn = GetChild(node, layerChild[res->name], false);

        if (!dn.empty()) {
            if (dn.child("crop").empty()) {
                for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {
                    pugi::xml_attribute attr = *ait;
                    res->params.emplace(attr.name(), attr.value());
                }
            } else {
                if (std::is_same<LT, CropLayer>::value) {
                    auto crop_res = std::dynamic_pointer_cast<CropLayer>(res);
                    if (!crop_res) {
                        THROW_IE_EXCEPTION << "Crop layer is nullptr";
                    }
                    std::string axisStr, offsetStr, dimStr;
                    FOREACH_CHILD(_cn, dn, "crop") {
                        int axis = GetIntAttr(_cn, "axis", 0);
                        crop_res->axis.push_back(axis);
                        axisStr += std::to_string(axis) + ",";
                        int offset = GetIntAttr(_cn, "offset", 0);
                        crop_res->offset.push_back(offset);
                        offsetStr += std::to_string(offset) + ",";
                    }
                    if (!axisStr.empty() && !offsetStr.empty() && !dimStr.empty()) {
                        res->params["axis"] = axisStr.substr(0, axisStr.size() - 1);
                        res->params["offset"] = offsetStr.substr(0, offsetStr.size() - 1);
                    }
                }
            }
        }
        return res;
    }

    std::map<std::string, std::vector<std::string>> layerChild;
};

class ActivationLayerCreator : public BaseCreator {
public:
    explicit ActivationLayerCreator(const std::string& type): BaseCreator(type) {}
    CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) override;
};

class TILayerCreator : public BaseCreator {
public:
    explicit TILayerCreator(const std::string& type): BaseCreator(type) {}
    CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) override;
};
}  // namespace details
}  // namespace InferenceEngine

/***********************************************************************************/
/******* End of Layer Parsers ******************************************************/
/***********************************************************************************/
