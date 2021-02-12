// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gna_plugin_log.hpp>
#include <cstdint>
#include <backend/dnn_types.h>
#include "gna_float_runtime.hpp"

using namespace GNAPluginNS;
using namespace GNAPluginNS::runtime;


void FP::infer() {
    if (!dnn) {
        THROW_GNA_EXCEPTION << "[GNA FP32 RUNTIME] not initialized";
    }

    for (uint32_t i = 0; i < dnn->component.size(); i++) {
        intel_dnn_component_t *comp = &dnn->component[i];
        uint32_t *ptr_active_outputs = nullptr;
        uint32_t num_active_outputs = (comp->orientation_out == kDnnInterleavedOrientation)
                                      ? comp->num_rows_out : comp->num_columns_out;

        if (i == dnn->component.size() - 1) {  // active list applies to last component
            ptr_active_outputs = dnn->ptr_active_outputs();
            num_active_outputs = dnn->num_active_outputs();
        } else if (i == dnn->component.size() - 2) {  // also applies to last two components when last is PWL
            if ((dnn->component[i].operation == kDnnAffineOp) && (dnn->component[i + 1].operation == kDnnPiecewiselinearOp)) {
                ptr_active_outputs = dnn->ptr_active_outputs();
                num_active_outputs = dnn->num_active_outputs();            }
        }

        switch (comp->operation) {
            case kDnnAffineOp : {
                ApplyAffineTransform(comp, ptr_active_outputs, num_active_outputs);
                break;
            }
            case kDnnDiagonalOp: {
                ApplyDiagonalTransform(comp);
                break;
            }
            case kDnnRecurrentOp: {
                if ((i < dnn->component.size() - 1) && (dnn->component[i + 1].operation == kDnnPiecewiselinearOp)) {
                    intel_dnn_component_t *comp_pwl = &dnn->component[i + 1];
                    for (uint32_t j = 0; j < comp->num_rows_in; j++) {
                        void *ptr_feedbacks =
                            reinterpret_cast<void *>(reinterpret_cast<int32_t *>(comp->op.recurrent.ptr_feedbacks)
                                + j * comp_pwl->num_columns_out);
                        ApplyRecurrentTransform(comp, j, ptr_feedbacks);
                        ApplyPiecewiseLinearTransform(comp_pwl, kDnnFloat, num_active_outputs, j);
                    }
                    i++;  // skip next component
                } else {
                    THROW_GNA_EXCEPTION << "Missing PiecewiseLinear component after Recurrent component in Propagate!";
                }
                break;
            }
            case kDnnConvolutional1dOp: {
                ApplyConvolutional1DTransform(comp);
                break;
            }
            case kDnnPiecewiselinearOp: {
                ApplyPiecewiseLinearTransform(comp, kDnnFloat, num_active_outputs);
                break;
            }
            case kDnnMaxPoolOp: {
                ApplyMaxPoolTransform(comp, kDnnFloat);
                break;
            }
            case kDnnInterleaveOp: {
                ApplyTranspose(comp);
                break;
            }
            case kDnnDeinterleaveOp: {
                ApplyTranspose(comp);
                break;
            }
            case kDnnCopyOp: {
                ApplyCopy(comp);
                break;
            }
            default:
                THROW_GNA_EXCEPTION << "[GNA FP32 RUNTIME] Bad operation " << comp->operation;
        }
    }
}