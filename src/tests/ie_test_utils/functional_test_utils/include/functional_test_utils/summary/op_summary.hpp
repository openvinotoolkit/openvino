// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "summary.hpp"

#include "openvino/opsets/opset.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace test {
namespace utils {

class OpSummary;

class OpSummaryDestroyer {
private:
    OpSummary *p_instance;
public:
    ~OpSummaryDestroyer();

    void initialize(OpSummary *p);
};

class OpSummary : public virtual Summary {
private:
    static OpSummary *p_instance;
    static bool extractBody;
    std::map<ov::NodeTypeInfo, PassRate> opsStats = {};

    std::string getOpVersion(const ov::NodeTypeInfo &type_info);

protected:
    OpSummary();
    static OpSummaryDestroyer destroyer;
    friend class OpSummaryDestroyer;

public:
    static OpSummary &getInstance();

    std::map<ov::NodeTypeInfo, PassRate> getOPsStats() { return opsStats; }

    static void setExtractBody(bool val) { extractBody = val; }
    static bool getExtractBody() { return extractBody; }

    std::map<std::string, PassRate> getStatisticFromReport();
    void saveReport() override;

    void updateOPsStats(const std::shared_ptr<ov::Model> &model, const PassRate::Statuses &status, double rel_influence_coef = 1);
    void updateOPsImplStatus(const std::shared_ptr<ov::Model> &model, const bool implStatus);

    void updateOPsStats(const ov::NodeTypeInfo &op, const PassRate::Statuses &status, double rel_influence_coef = 1);
    void updateOPsImplStatus(const ov::NodeTypeInfo &op, const bool implStatus);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
