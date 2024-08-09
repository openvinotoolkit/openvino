// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <mpi.h>
#include "oneapi/ccl.hpp"

namespace cldnn {

class Messenger {
private:
    Messenger() {
        atexit(Messenger::mpi_finalize);
        helperInit();
    }

    ~Messenger() {
        if (pcomm != nullptr) {
            helperFreePCOMM();
        }
    }

public:
    static Messenger &getInstance() {
        static Messenger instance;
        return instance;
    }

    int getRank() {
        return world_rank;
    }

    int getSize() {
        return world_size;
    }

private:
    Messenger(const Messenger &messenger) = delete;
    Messenger &operator=(const Messenger &messenger) = delete;

    static void mpi_finalize() {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);

        if (!is_finalized) {
            MPI_Finalize();
        }
    }

private:
    int world_size;
    int world_rank;
    int world_color;
    bool localRanksFlags;
    ccl::communicator *pcomm;

    int helperInit() {
        ccl::init();

        int is_initialized = 0;
        MPI_Initialized(&is_initialized);
        if (!is_initialized)
            MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // world_color = world_rank / tpSize = world_rank / (world_size / ppSize)
        // like: world_color = 0~7 / (8 / 4), XFT_PIPELINE_STAGE = ppSize = 4; tpSize = 2
        //       world_rank = 0, 1,  ->  world_color = ppRank = 0, 0,  ->  tpRank = 0, 1;
        //                    2, 3,                             1, 1,               0, 1;
        //                    4, 5,                             2, 2,               0, 1;
        //                    6, 7;                             3, 3;               0, 1;
        // world_color = world_rank / (world_size / world_color);
        // MPI_Comm row_comm;
        // MPI_Comm_split(MPI_COMM_WORLD, *world_color, *world_rank, &row_comm);

        // int row_size, row_rank;
        // MPI_Comm_size(row_comm, &row_size);
        // MPI_Comm_rank(row_comm, &row_rank);

        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type mainAddr;

        if (world_rank == 0) {
            kvs = ccl::create_main_kvs();
            mainAddr = kvs->get_address();
            MPI_Bcast(reinterpret_cast<void *>(mainAddr.data()), mainAddr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(reinterpret_cast<void *>(mainAddr.data()), mainAddr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(mainAddr);
        }

        pcomm = new ccl::communicator(ccl::create_communicator(world_size, world_rank, kvs));

        world_size = pcomm->size();
        world_rank = pcomm->rank();

        return 0;
    }

    void helperFreePCOMM() {
        delete pcomm;
    }

public:
    void helperAllgatherv(const  void *sendBuf, size_t count, void *recvBuf, const std::vector<long unsigned int> &recvCounts) {
        ccl::allgatherv(sendBuf, count, recvBuf, recvCounts, ccl::datatype::float32, *pcomm).wait();
    }

    void helperAllgathervBF16(const void* sendBuf, size_t count, void* recvBuf, const std::vector<long unsigned int> &recvCounts) {
        ccl::allgatherv(sendBuf, count, recvBuf, recvCounts, ccl::datatype::bfloat16, *pcomm).wait();
    }

    void helperAllreduce(void *sendBuf, void *recvBuf, size_t count) {
        ccl::allreduce(sendBuf, recvBuf, count, ccl::datatype::float32, ccl::reduction::sum, *pcomm).wait();
    }
    void helperAllreducef16(void *sendBuf, void *recvBuf, size_t count) {
        ccl::allreduce(sendBuf, recvBuf, count, ccl::datatype::float16, ccl::reduction::sum, *pcomm).wait();
    }
    void helperAllreduceBF16(void *sendBuf, void *recvBuf, size_t count) {
        ccl::allreduce(sendBuf, recvBuf, count, ccl::datatype::bfloat16, ccl::reduction::sum, *pcomm).wait();
    }
};
} // end namespace cldnn