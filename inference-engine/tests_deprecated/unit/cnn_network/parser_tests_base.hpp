// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <gtest/gtest.h>
#include "cnn_network_impl.hpp"
#include  <tests_common.hpp>
#include "ie_format_parser.h"
#include "ie_blob_proxy.hpp"
#include <string>
#include "pugixml.hpp"
#include "xml_parse_utils.h"
#include "mean_image.h"

#include "common_test_utils/xml_net_builder/xml_father.hpp"

class FormatParserTest : public TestsCommon {
public:
    FormatParserTest() {
    }

protected:
    const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
            '\\';
#else
            '/';
#endif
    const std::string parentDir = std::string("..") + std::to_string(FormatParserTest::kPathSeparator);

    std::string getParentDir(std::string currentFile) const {
        return parentDir + currentFile;
    }

protected:
    InferenceEngine::details::CNNNetworkImplPtr net;
    ModelsPath _path_to_models;

    InferenceEngine::InputInfo::Ptr getFirstInput() const {
        return ::getFirstInput(net.get());
    }

    template<class LayerType>
    std::shared_ptr<LayerType> getLayer(const std::string &name) const {
        InferenceEngine::CNNLayerPtr ptr;
        net->getLayerByName(name.c_str(), ptr, nullptr);
        return std::dynamic_pointer_cast<LayerType>(ptr);
    }


    virtual void SetUp() {
        _path_to_models += kPathSeparator;
    }

    void assertParseFail(const std::string &fileContent) {
        try {
            parse(fileContent);
            FAIL() << "Parser didn't throw";
        } catch (const std::exception &ex) {
            SUCCEED() << ex.what();
        }
    }

    void assertParseSucceed(const std::string &fileContent) {
        ASSERT_NO_THROW(parse(fileContent));
    }

    void assertSetWeightsFail(const InferenceEngine::TBlob<uint8_t>::Ptr &binBlob) {
        try {
            parser->SetWeights(binBlob);
            FAIL() << "Parser didn't throw";
        } catch (const std::exception &ex) {
            SUCCEED() << ex.what();
        }
    }

    void assertSetWeightsSucceed(const InferenceEngine::TBlob<uint8_t>::Ptr &binBlob) {
        ASSERT_NO_THROW(parser->SetWeights(binBlob));
    }

    void parse(const std::string &fileContent) {
        // check which version it is...
        pugi::xml_document xmlDoc;
        auto res = xmlDoc.load_string(fileContent.c_str());

        EXPECT_EQ(pugi::status_ok, res.status) << res.description() << " at offset " << res.offset;


        pugi::xml_node root = xmlDoc.document_element();

        int version = XMLParseUtils::GetIntAttr(root, "version", 2);
        if (version < 2) THROW_IE_EXCEPTION << "Deprecated IR's versions: " << version;
        if (version > 3) THROW_IE_EXCEPTION << "cannot parse future versions: " << version;
        parser.reset(new InferenceEngine::details::FormatParser(version));

        net = parser->Parse(root);
    }

#define initlayerIn(name, id, portid) \
    node("layer").attr("type", "Power").attr("name", name).attr("id", id)\
        .node("power_data").attr("power", 1).attr("scale", 1).attr("shift", 0).close()\
        .node("input")\
            .node("port").attr("id", portid)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()

#define initlayerInV2(name, id, portid) \
    node("layer").attr("type", "Power").attr("name", name).attr("id", id)\
        .node("power_data").attr("power", 1).attr("scale", 1).attr("shift", 0).close()\
        .node("input")\
            .node("port").attr("id", portid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()

#define initInputlayer(name, id, portid) \
    node("layer").attr("type", "Input").attr("name", name).attr("id", id)\
        .node("output")\
            .node("port").attr("id", portid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()

#define initInputlayer5D(name, id, portid) \
    node("layer").attr("type", "Input").attr("name", name).attr("id", id)\
        .node("output")\
            .node("port").attr("id", portid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_DEPTH)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()

#define initPowerlayerInOutV2(name, id, portid, outputid) \
    node("layer").attr("type", "Power").attr("name", name).attr("id", id)\
        .node("power_data").attr("power", 1).attr("scale", 1).attr("shift", 0).close()\
        .node("input")\
            .node("port").attr("id", portid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
        .node("output")\
            .node("port").attr("id", outputid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()


#define initPowerlayerInOut(name, id, portid, outputid) \
    node("layer").attr("type", "Power").attr("name", name).attr("id", id)\
        .node("power_data").attr("power", 1).attr("scale", 1).attr("shift", 0).close()\
        .node("input")\
            .node("port").attr("id", portid)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
        .node("output")\
            .node("port").attr("id", outputid)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
    .close()

#define initlayerInOut(name, type, id, portid, outputid) \
    node("layer").attr("type", type).attr("name", name).attr("id", id)\
        .node("input")\
            .node("port").attr("id", portid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
        .node("output")\
            .node("port").attr("id", outputid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\


#define initConv5DlayerInOut(name, id, group, output, kernel, pads_begin, pads_end, strides, dilations, inputid, outputid) \
    node("layer").attr("type", "Convolution").attr("name", name).attr("id", id)\
        .node("data").attr("group", group).attr("output", output).attr("kernel", kernel).attr("pads_begin", pads_begin).attr("pads_end", pads_end).attr("strides", strides).attr("dilations", dilations).close()\
        .node("input")\
            .node("port").attr("id", inputid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_DEPTH)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\
        .node("output")\
            .node("port").attr("id", outputid)\
              .node("dim", MT_BATCH)\
              .node("dim", MT_CHANNELS)\
              .node("dim", MT_DEPTH)\
              .node("dim", MT_HEIGHT)\
              .node("dim", MT_WIDTH)\
            .close()\
        .close()\


#define initedge(fl, fp, tl, tp)\
    node("edge").attr("from-layer", fl).attr("from-port", fp).attr("to-layer", tl).attr("to-port", tp).close()

/*
    auto net(const string & name1) const -> decltype(XMLFather().node("net").attr("name", name1).initlayers().initedges()) {
        return XMLFather().node("net").attr("name", name1).initlayers().initedges();
    }*/

#define MAKE_ALEXNET_FOR_MEAN_TESTS()\
xml().node("net").attr("name", "AlexNet")\
.node("input").attr("name", "data")\
    .node("dim", MT_CHANNELS)\
    .node("dim", MT_HEIGHT)\
    .node("dim", MT_WIDTH)\
    .newnode("layers")\
        .initPowerlayerInOut("power", 0, 0, 1)\
        .initlayerIn("power", 1, 0)\
    .newnode("edges")\
        .initedge(0,1,1,0)\
    .newnode("pre-process")

#define MAKE_ALEXNET_FOR_MEAN_TESTS_V2()\
xml().node("net").attr("name", "AlexNet").attr("version", 2)\
    .node("layers")\
        .initInputlayer("data", 0, 0)\
        .initPowerlayerInOutV2("power1", 1, 1, 2)\
        .initlayerInV2("power2", 2, 3)\
    .newnode("edges")\
        .initedge(0,0,1,1)\
        .initedge(1,2,2,3)\
    .newnode("pre-process")


#define BEGIN_NET()\
_BEGIN_NET(2)

#define BEGIN_NET_V3()\
_BEGIN_NET(3)

#define BEGIN_NET_V2()\
_BEGIN_NET(2)

#define _BEGIN_NET(x)\
xml().node("net").attr("name", "AlexNet").attr("version", x)\
    .node("layers")\
        .initInputlayer("data", 0, 0)\


#define END_NET()\
    .newnode("edges")\
        .initedge(0,0,1,1)\
            .close()


    template<class T>
    InferenceEngine::TBlob<uint8_t>::Ptr makeBinBlobForMeanTest() {
        typename InferenceEngine::TBlob<T>::Ptr binBlobFloat(
                new InferenceEngine::TBlob<T>({InferenceEngine::Precision::FP32,
                                               {MT_HEIGHT, MT_WIDTH, MT_CHANNELS}, InferenceEngine::CHW}));
        binBlobFloat->allocate();
        std::vector<T> meanValues = MeanImage<T>::getValue();
        std::copy(meanValues.begin(), meanValues.end(), (T *) binBlobFloat->data());
        InferenceEngine::SizeVector dims_dst = {MT_HEIGHT, MT_WIDTH * sizeof(T), MT_CHANNELS};
        typename InferenceEngine::TBlobProxy<uint8_t>::Ptr binBlob(new InferenceEngine::TBlobProxy<uint8_t>(
                InferenceEngine::Precision::FP32, InferenceEngine::CHW, binBlobFloat, 0, dims_dst));
        return binBlob;
    }

    template<class T>
    void assertMeanImagePerChannelCorrect() {
        std::vector<T> meanImage = MeanImage<T>::getValue();
        auto &pp = getFirstInput()->getPreProcess();
        ASSERT_EQ(MT_CHANNELS, pp.getNumberOfChannels());
        for (unsigned channel = 0, globalPixel = 0; channel < MT_CHANNELS; channel++) {
            auto actualMeanChannel = std::dynamic_pointer_cast<InferenceEngine::TBlob<T> >(pp[channel]->meanData);
            ASSERT_EQ(MT_HEIGHT * MT_WIDTH, actualMeanChannel->size());
            for (unsigned pixel = 0; pixel < actualMeanChannel->size(); pixel++, globalPixel++) {
                ASSERT_FLOAT_EQ(meanImage[globalPixel], actualMeanChannel->readOnly()[pixel]);
            }
        }
    }

    template<class T>
    void assertMeanImageCorrect() {
        std::vector<T> meanImage = MeanImage<T>::getValue();

        auto &pp = getFirstInput()->getPreProcess();
        ASSERT_EQ(MT_CHANNELS, pp.getNumberOfChannels());
        for (size_t c = 0; c < pp.getNumberOfChannels(); c++) {
            auto actualMeanTBlob = std::dynamic_pointer_cast<InferenceEngine::TBlob<T> >(pp[c]->meanData);
            ASSERT_EQ(MT_WIDTH, actualMeanTBlob->getTensorDesc().getDims().back());
            ASSERT_EQ(MT_HEIGHT,
                      actualMeanTBlob->getTensorDesc().getDims()[actualMeanTBlob->getTensorDesc().getDims().size() -
                                                                 2]);
            ASSERT_EQ(MT_WIDTH * MT_HEIGHT, actualMeanTBlob->size());
            for (unsigned index = 0; index < actualMeanTBlob->size(); index++) {
                ASSERT_FLOAT_EQ(meanImage[index + c * MT_WIDTH * MT_HEIGHT], actualMeanTBlob->readOnly()[index]);
            }
        }
    }

    CommonTestUtils::XMLFather xml() {
        return CommonTestUtils::XMLFather();
    }

    std::shared_ptr<InferenceEngine::details::FormatParser> parser;

public:

    int getXmlVersion(pugi::xml_node &root) {
        if (!root.child("InputData").empty()) return 2;
        return 1;
    }


    std::string getXmlPath(const std::string &filePath) {
        std::string xmlPath = filePath;
        const auto openFlags = std::ios_base::ate | std::ios_base::binary;
        std::ifstream fp(xmlPath, openFlags);
        //TODO: Dueto multi directory build systems, and single directory build system
        //, it is usualy a problem to deal with relative paths.
        if (!fp.is_open()) {
            fp.open(getParentDir(xmlPath), openFlags);
            EXPECT_TRUE(fp.is_open())
                                << "cannot open file " << xmlPath << " or " << getParentDir(xmlPath);
            fp.close();
            xmlPath = getParentDir(xmlPath);
        }
        return xmlPath;
    }

    std::string readFileContent(const std::string &filePath) {

        const auto openFlags = std::ios_base::ate | std::ios_base::binary;
        std::ifstream fp(getXmlPath(filePath), openFlags);
        EXPECT_TRUE(fp.is_open()) << "Cannot open file: " << filePath;
        if (!fp.is_open())
            return std::string();

        std::streamsize size = fp.tellg();
        EXPECT_GE(size, 1) << "file is empty: " << filePath;
        if (size == 0)
            return std::string();

        std::string str;

        str.reserve((size_t) size);
        fp.seekg(0, std::ios::beg);

        str.assign((std::istreambuf_iterator<char>(fp)),
                   std::istreambuf_iterator<char>());
        return str;
    }
};
