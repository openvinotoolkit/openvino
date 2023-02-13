// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include <pugixml.hpp>


#include "functional_test_utils/summary/op_summary.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace ov::test::utils;

#ifdef _WIN32
# define getpid _getpid
#endif

OpSummary *OpSummary::p_instance = nullptr;
bool OpSummary::extractBody = false;
OpSummaryDestroyer OpSummary::destroyer;

OpSummaryDestroyer::~OpSummaryDestroyer() {
    delete p_instance;
}

void OpSummaryDestroyer::initialize(OpSummary *p) {
    p_instance = p;
}

OpSummary::OpSummary() {
    reportFilename = CommonTestUtils::OP_REPORT_FILENAME;
}

OpSummary &OpSummary::getInstance() {
    if (!p_instance) {
        p_instance = new OpSummary();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

void OpSummary::updateOPsStats(const ov::NodeTypeInfo &op, const PassRate::Statuses &status) {
    auto it = opsStats.find(op);
    if (opsStats.find(op) == opsStats.end()) {
        opsStats.insert({op, PassRate()});
    }
    auto &passrate = opsStats[op];
    if (isCrashReported) {
        isCrashReported = false;
        if (passrate.crashed > 0)
            passrate.crashed--;
    }
    if (isHangReported) {
        isHangReported = false;
        return;
    }
    switch (status) {
        case PassRate::PASSED:
            if (!passrate.isImplemented) {
                passrate.isImplemented = true;
            }
            passrate.passed++;
            break;
        case PassRate::FAILED:
            passrate.failed++;
            break;
        case PassRate::SKIPPED:
            passrate.skipped++;
            break;
        case PassRate::CRASHED: {
            passrate.crashed++;
            isCrashReported = true;
            break;
        }
        case PassRate::HANGED: {
            passrate.hanged++;
            isHangReported = true;
            break;
        }
    }
}

void OpSummary::updateOPsImplStatus(const ov::NodeTypeInfo &op, const bool implStatus) {
    auto it = opsStats.find(op);
    if (it != opsStats.end()) {
        if (!it->second.isImplemented && implStatus) {
            it->second.isImplemented = true;
        }
    } else {
        opsStats[op] = PassRate();
        opsStats[op].isImplemented = implStatus;
    }
}

std::string OpSummary::getOpVersion(const ov::NodeTypeInfo &type_info) {
    std::string opset_name = "opset", version = type_info.get_version();
    auto pos = version.find(opset_name);
    if (pos == std::string::npos) {
        return "undefined";
    } else {
        return version.substr(pos + opset_name.size());
    }
}

std::map<std::string, PassRate> OpSummary::getStatisticFromReport() {
    pugi::xml_document doc;

    std::ifstream file;
    file.open(reportFilename);

    pugi::xml_node root;
    doc.load_file(reportFilename);
    root = doc.child("report");

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.child(deviceName.c_str());
    std::map<std::string, PassRate> oldOpsStat;
    for (auto &child : currentDeviceNode.children()) {
        std::string entry = child.name();
        auto p = std::stoi(child.attribute("passed").value());
        auto f = std::stoi(child.attribute("failed").value());
        auto s = std::stoi(child.attribute("skipped").value());
        auto c = std::stoi(child.attribute("crashed").value());
        auto h = std::stoi(child.attribute("hanged").value());
        PassRate obj(p, f, s, c, h);
        oldOpsStat.insert({entry, obj});
    }
    return oldOpsStat;
}

void OpSummary::updateOPsStats(const std::shared_ptr<ov::Model> &model, const PassRate::Statuses &status) {
    if (model->get_parameters().empty()) {
        return;
    }
    bool isFunctionalGraph = false, isReportConvert = true;
    for (const auto &op : model->get_ordered_ops()) {
        if (!std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) &&
            !std::dynamic_pointer_cast<ov::op::v0::Constant>(op) &&
            !std::dynamic_pointer_cast<ov::op::v0::Result>(op)) {
            // find all features
            if (!std::dynamic_pointer_cast<ov::op::v0::Convert>(op)) {
                isReportConvert = false;
            }
            isFunctionalGraph = true;
            if (!isReportConvert && isFunctionalGraph) {
                break;
            }
        }
    }

    for (const auto &op : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Constant>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Result>(op) || isFunctionalGraph) {
            continue;
        }
        if (!isReportConvert && std::dynamic_pointer_cast<ov::op::v0::Convert>(op)) {
            continue;
        }
        if (extractBody) {
            if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                updateOPsStats(ti_body, status);
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                updateOPsStats(loop_body, status);
            } else if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    updateOPsStats(if_body, status);
                }
            }
        }
        updateOPsStats(op->get_type_info(), status);
    }
}

void OpSummary::updateOPsImplStatus(const std::shared_ptr<ov::Model> &model, const bool implStatus) {
    if (model->get_parameters().empty()) {
        return;
    }
    bool isFunctionalGraph = false;
    for (const auto &op : model->get_ordered_ops()) {
        if (!std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) &&
            !std::dynamic_pointer_cast<ov::op::v0::Constant>(op) &&
            !std::dynamic_pointer_cast<ov::op::v0::Result>(op)) {
            isFunctionalGraph = true;
            break;
        }
    }

    for (const auto &op : model->get_ordered_ops()) {
        if ((std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
             std::dynamic_pointer_cast<ov::op::v0::Constant>(op) ||
             std::dynamic_pointer_cast<ov::op::v0::Result>(op)) && isFunctionalGraph) {
            continue;
        } else if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
            auto ti_body = ti->get_function();
            updateOPsImplStatus(ti_body, implStatus);
        } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
            auto loop_body = loop->get_function();
            updateOPsImplStatus(loop_body, implStatus);
        } else {
            updateOPsImplStatus(op->get_type_info(), implStatus);
        }
    }
}

#ifdef IE_TEST_DEBUG
void Summary::saveDebugReport(const char* className, const char* opName, unsigned long passed, unsigned long failed,
                              unsigned long skipped, unsigned long crashed, unsigned long hanged) {
    std::string outputFilePath = "./part_report.txt";
    std::ofstream file;
    file.open(outputFilePath, std::ios_base::app);
    file << className << ' ' << opName << ' ' << passed << ' ' << failed << ' ' << skipped << ' ' << crashed << ' ' << hanged << '\n';
    file.close();
}
#endif  //IE_TEST_DEBUG

void OpSummary::saveReport() {
    if (isReported) {
        return;
    }

    if (opsStats.empty()) {
        return;
    }

    std::string filename = reportFilename;
    if (saveReportWithUniqueName) {
        auto processId = std::to_string(getpid());
        filename += "_" + processId + "_" + ts;
    }
    filename += CommonTestUtils::REPORT_EXTENSION;

    if (!CommonTestUtils::directoryExists(outputFolder)) {
        CommonTestUtils::createDirectoryRecursive(outputFolder);
    }

    std::string outputFilePath = outputFolder + std::string(CommonTestUtils::FileSeparator) + filename;

    std::set<ov::NodeTypeInfo> opsInfo;
    for (const auto &opset_pair : get_available_opsets()) {
        std::string opset_version = opset_pair.first;
        const ov::OpSet& opset = opset_pair.second();
        const auto &type_info_set = opset.get_type_info_set();
        opsInfo.insert(type_info_set.begin(), type_info_set.end());
    }

    auto &summary = OpSummary::getInstance();
    auto stats = summary.getOPsStats();

    pugi::xml_document doc;

    const bool fileExists = CommonTestUtils::fileExists(outputFilePath);

    time_t rawtime;
    struct tm *timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime); // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (fileExists) {
        doc.load_file(outputFilePath.c_str());
        root = doc.child("report");
        //Ugly but shorter than to write predicate for find_atrribute() to update existing one
        root.remove_attribute("timestamp");
        root.append_attribute("timestamp").set_value(timeNow);

        root.remove_child("ops_list");
        root.child("results").remove_child(summary.deviceName.c_str());
    } else {
        root = doc.append_child("report");
        root.append_attribute("timestamp").set_value(timeNow);
        root.append_child("results");
    }

    pugi::xml_node opsNode = root.append_child("ops_list");
    for (const auto &op : opsInfo) {
        std::string name = std::string(op.name) + "-" + getOpVersion(op);
        pugi::xml_node entry = opsNode.append_child(name.c_str());
        (void) entry;
    }

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.append_child(summary.deviceName.c_str());
    std::unordered_set<std::string> opList;
    for (const auto &it : stats) {
        std::string name = std::string(it.first.name) + "-" + getOpVersion(it.first);
        opList.insert(name);
        pugi::xml_node entry = currentDeviceNode.append_child(name.c_str());
        entry.append_attribute("implemented").set_value(it.second.isImplemented);
        entry.append_attribute("passed").set_value(static_cast<unsigned long long>(it.second.passed));
        entry.append_attribute("failed").set_value(static_cast<unsigned long long>(it.second.failed));
        entry.append_attribute("skipped").set_value(static_cast<unsigned long long>(it.second.skipped));
        entry.append_attribute("crashed").set_value(static_cast<unsigned long long>(it.second.crashed));
        entry.append_attribute("hanged").set_value(static_cast<unsigned long long>(it.second.hanged));
        entry.append_attribute("passrate").set_value(it.second.getPassrate());
    }

    if (extendReport && fileExists) {
        auto opStataFromReport = summary.getStatisticFromReport();
        for (auto &item : opStataFromReport) {
            pugi::xml_node entry;
            if (opList.find(item.first) == opList.end()) {
                entry = currentDeviceNode.append_child(item.first.c_str());
                entry.append_attribute("implemented").set_value(item.second.isImplemented);
                entry.append_attribute("passed").set_value(static_cast<unsigned long long>(item.second.passed));
                entry.append_attribute("failed").set_value(static_cast<unsigned long long>(item.second.failed));
                entry.append_attribute("skipped").set_value(static_cast<unsigned long long>(item.second.skipped));
                entry.append_attribute("crashed").set_value(static_cast<unsigned long long>(item.second.crashed));
                entry.append_attribute("hanged").set_value(static_cast<unsigned long long>(item.second.hanged));
                entry.append_attribute("passrate").set_value(item.second.getPassrate());
            } else {
                entry = currentDeviceNode.child(item.first.c_str());
                auto implStatus = entry.attribute("implemented").value() == std::string("true") ? true : false;
                auto p = std::stoi(entry.attribute("passed").value()) + item.second.passed;
                auto f = std::stoi(entry.attribute("failed").value()) + item.second.failed;
                auto s = std::stoi(entry.attribute("skipped").value()) + item.second.skipped;
                auto c = std::stoi(entry.attribute("crashed").value()) + item.second.crashed;
                auto h = std::stoi(entry.attribute("hanged").value()) + item.second.hanged;
                PassRate obj(p, f, s, c, h);

                (implStatus || obj.isImplemented)
                ? entry.attribute("implemented").set_value(true)
                : entry.attribute("implemented").set_value(false);
                entry.attribute("passed").set_value(static_cast<unsigned long long>(obj.passed));
                entry.attribute("failed").set_value(static_cast<unsigned long long>(obj.failed));
                entry.attribute("skipped").set_value(static_cast<unsigned long long>(obj.skipped));
                entry.attribute("crashed").set_value(static_cast<unsigned long long>(obj.crashed));
                entry.attribute("hanged").set_value(static_cast<unsigned long long>(obj.hanged));
                entry.attribute("passrate").set_value(obj.getPassrate());
            }
        }
    }

    auto exitTime = std::chrono::system_clock::now() + std::chrono::seconds(saveReportTimeout);
    bool result = false;
    do {
        result = doc.save_file(outputFilePath.c_str());
    } while (!result && std::chrono::system_clock::now() < exitTime);

    if (!result) {
        std::string errMessage = "Failed to write report to " + outputFilePath;
        throw std::runtime_error(errMessage);
    } else {
        isReported = true;
    }
}
