// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ie_blob.h"
#include "blob_factory.hpp"
#include "utils/blob_dump.h"
#include <cpp/ie_cnn_net_reader.h>

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

TEST(MKLDNNDumpTests, UnallocatedBlob_NoDump) {
    SizeVector dims {2,3,4,5};
    Blob::Ptr blob = make_blob_with_precision({Precision::U8, dims, NHWC});

    std::stringstream buff;

    EXPECT_THROW({
        BlobDumper(blob).dump(buff);
    }, details::InferenceEngineException);
}

TEST(MKLDNNDumpTests, EmptyBlob_NoDump) {
    SizeVector dims {2,3,4,5};
    Blob::Ptr blob;

    std::stringstream buff;

    EXPECT_THROW({
        BlobDumper(blob).dump(buff);
    }, details::InferenceEngineException);
}

TEST(MKLDNNDumpTests, Ser) {
    SizeVector dims {2,3,4,5};
    Blob::Ptr blob = make_blob_with_precision({Precision::U8, dims, NHWC});
    blob->allocate();

    std::stringstream buff;
    BlobDumper(blob).dump(buff);

    ASSERT_GT(buff.str().size(), blob->byteSize());
}

TEST(MKLDNNDumpTests, SerDeser) {
    SizeVector dims {2,3,4,5};
    Blob::Ptr blob = make_blob_with_precision({Precision::U8, dims, NCHW});
    blob->allocate();

    std::stringstream buff;

    BlobDumper(blob).dump(buff);
    Blob::Ptr deser_blob = BlobDumper::read(buff).get();

    ASSERT_EQ(deser_blob->getTensorDesc().getDims(), blob->getTensorDesc().getDims());
    ASSERT_EQ(deser_blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getPrecision());

    std::vector<uint8_t> data(blob->buffer().as<uint8_t*>(), blob->buffer().as<uint8_t*>() + blob->size());
    std::vector<uint8_t> deser_data(deser_blob->buffer().as<uint8_t*>(), deser_blob->buffer().as<uint8_t*>()
                                    + deser_blob->size());
    ASSERT_EQ(deser_data, data);
}

TEST(MKLDNNDumpTests, SerDeserWithScales) {
    SizeVector dims {2,3,4,5};
    auto blob = make_blob_with_precision({Precision::U8, dims, NCHW});
    blob->allocate();

    auto scls = make_blob_with_precision({Precision::FP32, {3}, C});
    scls->allocate();

    std::stringstream buff;

    BlobDumper(blob).withScales(scls).dump(buff);
    auto deser = BlobDumper::read(buff);
    auto deser_blob = deser.get();
    auto deser_scls = deser.getScales();

    ASSERT_EQ(deser_blob->getTensorDesc().getDims(), blob->getTensorDesc().getDims());
    ASSERT_EQ(deser_blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getPrecision());

    std::vector<uint8_t> data(blob->buffer().as<uint8_t*>(), blob->buffer().as<uint8_t*>() + blob->size());
    std::vector<uint8_t> deser_data(deser_blob->buffer().as<uint8_t*>(), deser_blob->buffer().as<uint8_t*>()
                                                                         + deser_blob->size());
    ASSERT_EQ(deser_data, data);

    std::vector<uint8_t> scls_data(scls->buffer().as<uint8_t*>(), scls->buffer().as<uint8_t*>() + scls->size());
    std::vector<uint8_t> deser_scls_data(deser_scls->buffer().as<uint8_t*>(), deser_scls->buffer().as<uint8_t*>()
                                                                         + deser_scls->size());
    ASSERT_EQ(deser_scls_data, scls_data);
}


TEST(MKLDNNDumpTests, SerU8AsTxt) {
    SizeVector dims {2,3,4,5};

    Blob::Ptr blob = make_blob_with_precision({Precision::U8, dims, NCHW});
    blob->allocate();

    Blob::Ptr scls = make_blob_with_precision({Precision::FP32, {dims[1]}, C});
    scls->allocate();

    std::stringstream buff;
    BlobDumper(blob).withScales(scls).dumpAsTxt(buff);

    std::string deser_header, ref_header = "U8 4D shape: 2 3 4 5 (120)";
    std::getline(buff, deser_header);
    ASSERT_EQ(deser_header, ref_header);

    auto num_line = std::count(std::istreambuf_iterator<char>(buff),
            std::istreambuf_iterator<char>(), '\n');
    ASSERT_EQ(num_line, blob->size());
}

TEST(MKLDNNDumpTests, SerAsTxt) {
    SizeVector dims {2,3};

    Blob::Ptr blob = make_blob_with_precision({Precision::FP32, dims, NC});
    blob->allocate();

    Blob::Ptr scls = make_blob_with_precision({Precision::FP32, {dims[1]}, C});
    scls->allocate();

    std::stringstream buff;
    BlobDumper(blob).withScales(scls).dumpAsTxt(buff);

    std::string deser_header, ref_header = "FP32 2D shape: 2 3 (6)";
    std::getline(buff, deser_header);
    ASSERT_EQ(deser_header, ref_header);

    auto num_line = std::count(std::istreambuf_iterator<char>(buff),
                               std::istreambuf_iterator<char>(), '\n');
    ASSERT_EQ(num_line, blob->size());
}