// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/op_summary.hpp"

#include <algorithm>
#include <pugixml.hpp>

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/summary/op_info.hpp"

using namespace ov::test::utils;

#ifdef _WIN32
#    define getpid _getpid
#endif

OpSummary* OpSummary::p_instance = nullptr;
bool OpSummary::extractBody = false;
OpSummaryDestroyer OpSummary::destroyer;

OpSummaryDestroyer::~OpSummaryDestroyer() {
    delete p_instance;
}

void OpSummaryDestroyer::initialize(OpSummary* p) {
    p_instance = p;
}

OpSummary::OpSummary() {
    reportFilename = ov::test::utils::OP_REPORT_FILENAME;
}

OpSummary& OpSummary::createInstance() {
    if (!p_instance) {
        p_instance = new OpSummary();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

OpSummary& OpSummary::getInstance() {
    return createInstance();
}

void OpSummary::updateOPsStats(const ov::NodeTypeInfo& op,
                               const PassRate::Statuses& status,
                               double rel_influence_coef) {
    auto it = opsStats.find(op);
    if (opsStats.find(op) == opsStats.end()) {
        opsStats.insert({op, PassRate()});
    }
    auto& passrate = opsStats[op];
    if (passrate.isCrashReported) {
        passrate.isCrashReported = false;
        if (passrate.crashed > 0)
            passrate.crashed--;
    } else {
        passrate.rel_all += rel_influence_coef;
    }
    if (passrate.isHangReported) {
        passrate.isHangReported = false;
        return;
    }
    switch (status) {
    case PassRate::PASSED:
        if (!passrate.isImplemented) {
            passrate.isImplemented = true;
        }
        passrate.passed++;
        passrate.rel_passed += rel_influence_coef;
        break;
    case PassRate::FAILED:
        passrate.failed++;
        break;
    case PassRate::SKIPPED:
        passrate.skipped++;
        break;
    case PassRate::CRASHED: {
        passrate.crashed++;
        passrate.isCrashReported = true;
        break;
    }
    case PassRate::HANGED: {
        passrate.hanged++;
        passrate.isHangReported = true;
        break;
    }
    }
}

void OpSummary::updateOPsImplStatus(const ov::NodeTypeInfo& op, const bool implStatus) {
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

std::string OpSummary::get_opset_number(const std::string& opset_full_name) {
    std::string opset_name = "opset";
    auto pos = opset_full_name.find(opset_name);
    if (pos == std::string::npos) {
        return "undefined";
    } else {
        return opset_full_name.substr(pos + opset_name.size());
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
    for (auto& child : currentDeviceNode.children()) {
        std::string entry = child.name();
        auto p = std::stoi(child.attribute("passed").value());
        auto f = std::stoi(child.attribute("failed").value());
        auto s = std::stoi(child.attribute("skipped").value());
        auto c = std::stoi(child.attribute("crashed").value());
        auto h = std::stoi(child.attribute("hanged").value());
        auto rel_passed = std::stoi(child.attribute("rel_passed").value());
        auto rel_all = std::stoi(child.attribute("rel_all").value());
        PassRate obj(p, f, s, c, h, rel_passed, rel_all);
        oldOpsStat.insert({entry, obj});
    }
    return oldOpsStat;
}

void OpSummary::updateOPsStats(const std::shared_ptr<ov::Model>& model, const PassRate::Statuses& status, double k) {
    bool isFunctionalGraph = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::as_type_ptr<ov::op::v0::Parameter>(op) && !ov::as_type_ptr<ov::op::v0::Constant>(op) &&
            !ov::as_type_ptr<ov::op::v0::Result>(op)) {
            // find all features
            isFunctionalGraph = true;
            break;
        }
    }

    for (const auto& op : model->get_ordered_ops()) {
        if ((ov::as_type_ptr<ov::op::v0::Parameter>(op) || ov::as_type_ptr<ov::op::v0::Constant>(op) ||
             ov::as_type_ptr<ov::op::v0::Result>(op)) &&
            isFunctionalGraph) {
            continue;
        }
        if (extractBody) {
            if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
                updateOPsStats(op->get_type_info(), status, k);
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                updateOPsStats(ti_body, status, k);
            } else if (ov::as_type_ptr<ov::op::v5::Loop>(op)) {
                updateOPsStats(op->get_type_info(), status, k);
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                updateOPsStats(loop_body, status, k);
            } else if (ov::as_type_ptr<ov::op::v8::If>(op)) {
                updateOPsStats(op->get_type_info(), status, k);
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    updateOPsStats(if_body, status, k);
                }
            }
        }
        updateOPsStats(op->get_type_info(), status, k);
    }
}

void OpSummary::updateOPsImplStatus(const std::shared_ptr<ov::Model>& model, const bool implStatus) {
    if (model->get_parameters().empty()) {
        return;
    }
    bool isFunctionalGraph = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::as_type_ptr<ov::op::v0::Parameter>(op) && !ov::as_type_ptr<ov::op::v0::Constant>(op) &&
            !ov::as_type_ptr<ov::op::v0::Result>(op)) {
            isFunctionalGraph = true;
            break;
        }
    }

    for (const auto& op : model->get_ordered_ops()) {
        if ((ov::as_type_ptr<ov::op::v0::Parameter>(op) || ov::as_type_ptr<ov::op::v0::Constant>(op) ||
             ov::as_type_ptr<ov::op::v0::Result>(op)) &&
            isFunctionalGraph) {
            continue;
        } else if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
            auto ti_body = ti->get_function();
            updateOPsImplStatus(ti_body, implStatus);
        } else if (ov::as_type_ptr<ov::op::v5::Loop>(op)) {
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
            auto loop_body = loop->get_function();
            updateOPsImplStatus(loop_body, implStatus);
        } else {
            updateOPsImplStatus(op->get_type_info(), implStatus);
        }
    }
}

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
    filename += ov::test::utils::REPORT_EXTENSION;

    if (!ov::util::directory_exists(outputFolder)) {
        ov::util::create_directory_recursive(outputFolder);
    }

    std::string outputFilePath = outputFolder + std::string(ov::test::utils::FileSeparator) + filename;

    std::map<ov::NodeTypeInfo, std::string> opsInfo;
    for (const auto& opset_pair : get_available_opsets()) {
        std::string opset_version = opset_pair.first;
        const ov::OpSet& opset = opset_pair.second();
        const auto& type_info_set = opset.get_type_info_set();
        for (const auto& type_info : type_info_set) {
            auto it = opsInfo.find(type_info);
            std::string op_version = get_opset_number(opset_version);
            if (it == opsInfo.end()) {
                opsInfo.insert({type_info, op_version});
            } else {
                opsInfo[type_info] += " ";
                opsInfo[type_info] += op_version;
            }
        }
    }

    auto& summary = OpSummary::getInstance();
    auto stats = summary.getOPsStats();

    pugi::xml_document doc;

    const bool fileExists = ov::test::utils::fileExists(outputFilePath);

    time_t rawtime;
    struct tm* timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime);  // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (fileExists) {
        doc.load_file(outputFilePath.c_str());
        root = doc.child("report");
        // Ugly but shorter than to write predicate for find_atrribute() to update existing one
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
    for (const auto& op : opsInfo) {
        std::string name = functional::get_node_version(op.first);
        opsNode.append_child(name.c_str()).append_attribute("opsets").set_value(op.second.c_str());
    }

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.append_child(summary.deviceName.c_str());
    std::unordered_set<std::string> opList;
    for (auto& it : stats) {
        std::string name = functional::get_node_version(it.first);
        opList.insert(name);
        pugi::xml_node entry = currentDeviceNode.append_child(name.c_str());
        entry.append_attribute("implemented").set_value(it.second.isImplemented);
        entry.append_attribute("passed").set_value(static_cast<unsigned long long>(it.second.passed));
        entry.append_attribute("failed").set_value(static_cast<unsigned long long>(it.second.failed));
        entry.append_attribute("skipped").set_value(static_cast<unsigned long long>(it.second.skipped));
        entry.append_attribute("crashed").set_value(static_cast<unsigned long long>(it.second.crashed));
        entry.append_attribute("hanged").set_value(static_cast<unsigned long long>(it.second.hanged));
        entry.append_attribute("passrate").set_value(it.second.getPassrate());
        entry.append_attribute("relative_passed").set_value(it.second.rel_passed);
        entry.append_attribute("relative_all").set_value(it.second.rel_all);
        entry.append_attribute("relative_passrate").set_value(it.second.getRelPassrate());
    }

    if (extendReport && fileExists) {
        auto opStataFromReport = summary.getStatisticFromReport();
        for (auto& item : opStataFromReport) {
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
                entry.append_attribute("relative_passed").set_value(item.second.rel_passed);
                entry.append_attribute("relative_all").set_value(item.second.rel_all);
                entry.append_attribute("relative_passrate").set_value(item.second.getRelPassrate());
            } else {
                entry = currentDeviceNode.child(item.first.c_str());
                auto implStatus = entry.attribute("implemented").value() == std::string("true") ? true : false;
                auto p = std::stoi(entry.attribute("passed").value()) + item.second.passed;
                auto f = std::stoi(entry.attribute("failed").value()) + item.second.failed;
                auto s = std::stoi(entry.attribute("skipped").value()) + item.second.skipped;
                auto c = std::stoi(entry.attribute("crashed").value()) + item.second.crashed;
                auto h = std::stoi(entry.attribute("hanged").value()) + item.second.hanged;
                auto rel_passed = std::stoi(entry.attribute("relative_passed").value()) + item.second.rel_passed;
                auto rel_all = std::stoi(entry.attribute("relative_all").value()) + item.second.rel_all;
                PassRate obj(p, f, s, c, h, rel_passed, rel_all);

                (implStatus || obj.isImplemented) ? entry.attribute("implemented").set_value(true)
                                                  : entry.attribute("implemented").set_value(false);
                entry.attribute("passed").set_value(static_cast<unsigned long long>(obj.passed));
                entry.attribute("failed").set_value(static_cast<unsigned long long>(obj.failed));
                entry.attribute("skipped").set_value(static_cast<unsigned long long>(obj.skipped));
                entry.attribute("crashed").set_value(static_cast<unsigned long long>(obj.crashed));
                entry.attribute("hanged").set_value(static_cast<unsigned long long>(obj.hanged));
                entry.attribute("passrate").set_value(obj.getPassrate());
                entry.attribute("relative_passed").set_value(item.second.rel_passed);
                entry.attribute("relative_all").set_value(item.second.rel_all);
                entry.attribute("relative_passrate").set_value(item.second.getRelPassrate());
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
