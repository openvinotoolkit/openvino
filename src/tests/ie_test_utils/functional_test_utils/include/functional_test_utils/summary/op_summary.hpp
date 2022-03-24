// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "summary.hpp"

namespace LayerTestsUtils {

class Summary;
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
    std::vector<ngraph::OpSet> opsets;
    std::map<ngraph::NodeTypeInfo, PassRate> opsStats = {};

    std::string getOpVersion(const ngraph::NodeTypeInfo &type_info);

protected:
    static OpSummaryDestroyer destroyer;
    OpSummary();

    friend class OpSummaryDestroyer;

public:
    static OpSummary &getInstance();

    std::map<ngraph::NodeTypeInfo, PassRate> getOPsStats() { return opsStats; }

    std::vector<ngraph::OpSet> getOpSets() {
        return opsets;
    }

    static void setExtractBody(bool val) { extractBody = val; }
    static bool getExtractBody() { return extractBody; }

    std::map<std::string, PassRate> getStatisticFromReport() override;
    void saveReport() override;


    void updateOPsStats(const std::shared_ptr<ngraph::Function> &function, const PassRate::Statuses &status);
    void updateOPsImplStatus(const std::shared_ptr<ngraph::Function> &function, const bool implStatus);

    void updateOPsStats(const ngraph::NodeTypeInfo &op, const PassRate::Statuses &status);
    void updateOPsImplStatus(const ngraph::NodeTypeInfo &op, const bool implStatus);
};

}  // namespace LayerTestsUtils
