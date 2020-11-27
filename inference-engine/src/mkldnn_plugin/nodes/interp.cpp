// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include "ie_parallel.hpp"
#include "jit_generator.hpp"

using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

#define GET_OFF(field) offsetof(jit_args_interp, field)

struct jit_args_interp {
    const float *src00;
    const float *src01;
    const float *src10;
    const float *src11;
    float *dst;
    float *h_lambda0;
    float *h_lambda1;
    float *w_lambda0;
    float *w_lambda1;
};

struct jit_uni_interp_kernel {
    void (*ker_)(const jit_args_interp *);

    void operator()(const jit_args_interp *args) { assert(ker_); ker_(args); }

    jit_uni_interp_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_interp_kernel() {}
};

template <cpu_isa_t isa>
struct jit_uni_interp_kernel_f32 : public jit_uni_interp_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interp_kernel_f32)

    jit_uni_interp_kernel_f32() : jit_uni_interp_kernel(), jit_generator() {
        this->preamble();

        mov(reg_src00, ptr[reg_params + GET_OFF(src00)]);
        mov(reg_src01, ptr[reg_params + GET_OFF(src01)]);
        mov(reg_src10, ptr[reg_params + GET_OFF(src10)]);
        mov(reg_src11, ptr[reg_params + GET_OFF(src11)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_h_lambda0, ptr[reg_params + GET_OFF(h_lambda0)]);
        mov(reg_h_lambda1, ptr[reg_params + GET_OFF(h_lambda1)]);
        mov(reg_w_lambda0, ptr[reg_params + GET_OFF(w_lambda0)]);
        mov(reg_w_lambda1, ptr[reg_params + GET_OFF(w_lambda1)]);

        uni_vmovups(vmm_src00, ptr[reg_src00]);
        uni_vmovups(vmm_src01, ptr[reg_src01]);
        uni_vmovups(vmm_src10, ptr[reg_src10]);
        uni_vmovups(vmm_src11, ptr[reg_src11]);

        uni_vbroadcastss(vmm_h_lambda0, ptr[reg_h_lambda0]);
        uni_vbroadcastss(vmm_h_lambda1, ptr[reg_h_lambda1]);
        uni_vbroadcastss(vmm_w_lambda0, ptr[reg_w_lambda0]);
        uni_vbroadcastss(vmm_w_lambda1, ptr[reg_w_lambda1]);

        if (isa != sse42) {
            uni_vmulps(vmm_src01, vmm_src01, vmm_w_lambda0);
            uni_vmulps(vmm_src11, vmm_src11, vmm_w_lambda0);
            uni_vfmadd231ps(vmm_src01, vmm_w_lambda1, vmm_src00);
            uni_vfmadd231ps(vmm_src11, vmm_w_lambda1, vmm_src10);
            uni_vmulps(vmm_src01, vmm_src01, vmm_h_lambda1);
            uni_vfmadd231ps(vmm_src01, vmm_h_lambda0, vmm_src11);
            uni_vmovups(ptr[reg_dst], vmm_src01);
        } else {
            uni_vmulps(vmm_src01, vmm_src01, vmm_w_lambda0);
            uni_vmulps(vmm_src11, vmm_src11, vmm_w_lambda0);
            uni_vfmadd231ps(vmm_src01, vmm_w_lambda1, vmm_src00);
            // uni_vfmadd231ps affects XMM (vmm_w_lambda1) register. Need to initialize again.
            uni_vbroadcastss(vmm_w_lambda1, ptr[reg_w_lambda1]);
            uni_vfmadd231ps(vmm_src11, vmm_w_lambda1, vmm_src10);
            uni_vmulps(vmm_src01, vmm_src01, vmm_h_lambda1);
            uni_vfmadd231ps(vmm_src01, vmm_h_lambda0, vmm_src11);
            uni_vmovups(ptr[reg_dst], vmm_src01);

            // Next 4 elements
            size_t stride = 4 * sizeof(float);

            add(reg_src00, stride);
            add(reg_src01, stride);
            add(reg_src10, stride);
            add(reg_src11, stride);
            add(reg_dst, stride);

            uni_vmovups(vmm_src00, ptr[reg_src00]);
            uni_vmovups(vmm_src01, ptr[reg_src01]);
            uni_vmovups(vmm_src10, ptr[reg_src10]);
            uni_vmovups(vmm_src11, ptr[reg_src11]);

            uni_vbroadcastss(vmm_h_lambda0, ptr[reg_h_lambda0]);
            uni_vbroadcastss(vmm_w_lambda1, ptr[reg_w_lambda1]);

            uni_vmulps(vmm_src01, vmm_src01, vmm_w_lambda0);
            uni_vmulps(vmm_src11, vmm_src11, vmm_w_lambda0);
            uni_vfmadd231ps(vmm_src01, vmm_w_lambda1, vmm_src00);
            uni_vbroadcastss(vmm_w_lambda1, ptr[reg_w_lambda1]);
            uni_vfmadd231ps(vmm_src11, vmm_w_lambda1, vmm_src10);
            uni_vmulps(vmm_src01, vmm_src01, vmm_h_lambda1);
            uni_vfmadd231ps(vmm_src01, vmm_h_lambda0, vmm_src11);
            uni_vmovups(ptr[reg_dst], vmm_src01);
        }

        this->postamble();
        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == sse42, Xbyak::Xmm, isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src00 = r8;
    Xbyak::Reg64 reg_src01 = r9;
    Xbyak::Reg64 reg_src10 = r10;
    Xbyak::Reg64 reg_src11 = r11;
    Xbyak::Reg64 reg_dst   = rbp;
    Xbyak::Reg64 reg_h_lambda0 = r12;
    Xbyak::Reg64 reg_h_lambda1 = r13;
    Xbyak::Reg64 reg_w_lambda0 = r14;
    Xbyak::Reg64 reg_w_lambda1 = r15;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_src00 = Vmm(0);
    Vmm vmm_src01 = Vmm(1);
    Vmm vmm_src10 = Vmm(2);
    Vmm vmm_src11 = Vmm(3);
    Vmm vmm_h_lambda0 = Vmm(4);
    Vmm vmm_h_lambda1 = Vmm(5);
    Vmm vmm_w_lambda0 = Vmm(6);
    Vmm vmm_w_lambda1 = Vmm(7);
    Vmm vmm_dst   = Vmm(8);
};

class InterpImpl: public ExtLayerBase {
public:
    explicit InterpImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            auto inData = layer->insData[0].lock();
            if (inData == nullptr) {
                THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has nullable input data.";
            }
            if (inData->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "Interp supports only 4d blobs!";

            // We don't read other parameters since they are needed only for dst reshape in caffe
            pad_beg = layer->GetParamAsInt("pad_beg");
            pad_end = layer->GetParamAsInt("pad_end");
            align_corners = layer->GetParamAsBool("align_corners", true);

            ConfLayout blk_layout;
            if (inData->getTensorDesc().getPrecision() == Precision::U8) {
                LayerConfig config;
                DataConfig dataConfigDct;
                dataConfigDct.desc = TensorDesc(Precision::U8, inData->getTensorDesc().getDims(), Layout::NCHW);
                config.inConfs.push_back(dataConfigDct);

                DataConfig dataConfigOut;
                const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
                SizeVector blocks = out_dims;
                SizeVector order(blocks.size());
                SizeVector dimOffsets(blocks.size());
                SizeVector strides(blocks.size());
                size_t offset((std::numeric_limits<size_t>::max)());
                for (size_t i = 0; i < order.size(); i++) {
                    strides[i] = (std::numeric_limits<size_t>::max)();
                    dimOffsets[i] = 0;
                    order[i] = i;
                }
                dataConfigOut.desc = TensorDesc(Precision::FP32, out_dims, { blocks, order, offset, dimOffsets, strides });
                config.outConfs.push_back(dataConfigOut);
                config.dynBatchSupport = false;
                confs.push_back(config);
            } else {
                if (mayiuse(avx512_common)) {
                    blk_layout = ConfLayout::BLK16;
                    interp_kernel.reset(new jit_uni_interp_kernel_f32<avx512_common>());
                    addConfig(layer, { DataConfigurator(blk_layout, Precision::FP32) }, { DataConfigurator(blk_layout, Precision::FP32) });
                } else if (mayiuse(avx2)) {
                    blk_layout = ConfLayout::BLK8;
                    interp_kernel.reset(new jit_uni_interp_kernel_f32<avx2>());
                    addConfig(layer, { DataConfigurator(blk_layout, Precision::FP32) }, { DataConfigurator(blk_layout, Precision::FP32) });
                } else {
                    blk_layout = ConfLayout::BLK8;
                    interp_kernel.reset(new jit_uni_interp_kernel_f32<sse42>());
                    addConfig(layer, { DataConfigurator(blk_layout, Precision::FP32) }, { DataConfigurator(blk_layout, Precision::FP32) });
                }
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1) {
            strncpy(resp->msg, "Interp layer has invalid configs", sizeof(resp->msg));
            return GENERAL_ERROR;
        }

        if (config.inConfs[0].desc.getDims().size() != 4) {
            std::ostringstream result;
            result << "Interp layer has invalid layout: " << config.inConfs[0].desc.getLayout();
            strncpy(resp->msg, result.str().c_str(), sizeof(resp->msg) - 1);
            return GENERAL_ERROR;
        }

        auto inPrecision = config.inConfs[0].desc.getPrecision();
        if (inPrecision != Precision::U8 && inPrecision != Precision::FP32)  {
            strncpy(resp->msg, "Interp layer has unsupported input precision", sizeof(resp->msg));
            return GENERAL_ERROR;
        }

        if (config.outConfs[0].desc.getPrecision() != Precision::FP32)  {
            strncpy(resp->msg, "Interp layer has unsupported output precision", sizeof(resp->msg));
            return GENERAL_ERROR;
        }

        return OK;
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
#ifdef WIN32
#undef IN
#endif
        size_t IN = inputs[0]->getTensorDesc().getDims()[0];
        size_t IH = inputs[0]->getTensorDesc().getDims()[2];
        size_t IW = inputs[0]->getTensorDesc().getDims()[3];
        size_t OH = outputs[0]->getTensorDesc().getDims()[2];
        size_t OW = outputs[0]->getTensorDesc().getDims()[3];

        size_t IH_pad = IH + pad_beg + pad_end;
        size_t IW_pad = IW + pad_beg + pad_end;

        auto *dst_data = outputs[0]->buffer().as<float *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        {
            const float* src_data = inputs[0]->cbuffer().as<const float *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            size_t IC = (inputs[0]->getTensorDesc().getLayout() == Layout::BLOCKED)
                ? inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1] *
                inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[4]
                : IC = inputs[0]->getTensorDesc().getDims()[1];
            interpolate(IN, IC, src_data,
                -pad_beg, -pad_beg, IH_pad, IW_pad, IH, IW, dst_data, 0, 0, OH, OW, OH, OW);
        }
        break;
        case Precision::U8:
        {
            const uint8_t* src_data = inputs[0]->cbuffer().as<const uint8_t *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            size_t IC = inputs[0]->getTensorDesc().getDims()[1];
            interpolate_8u(inputs[0]->getTensorDesc().getLayout(), IN, IC, src_data,
                -pad_beg, -pad_beg, IH_pad, IW_pad, IH, IW, dst_data, 0, 0, OH, OW, OH, OW);
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect input precision. Only U8 or FP32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    int pad_beg;
    int pad_end;
    bool align_corners;
    std::shared_ptr<jit_uni_interp_kernel> interp_kernel;

    void interpolate(const size_t N, const size_t C,
                     const float *src, const int x1, const int y1,
                     const int IH_pad, const int IW_pad, const size_t IH, const size_t IW,
                     float *dst, const int x2, const int y2,
                     const int OH_pad, const int OW_pad, const size_t OH, const size_t OW) {
        if (IH_pad == OH_pad && IW_pad == OW_pad) {
            for (size_t i = 0; i < N * C * OH * OW; i++) {
                dst[i] = src[i];
            }
            return;
        }

        float rh;
        float rw;
        if (align_corners) {
            rh = (OH_pad > 1) ? static_cast<float>(IH_pad - 1) / (OH_pad - 1) : 0.0f;
            rw = (OW_pad > 1) ? static_cast<float>(IW_pad - 1) / (OW_pad - 1) : 0.0f;
        } else {
            rh = static_cast<float>(IH_pad) / (OH_pad);
            rw = static_cast<float>(IW_pad) / (OW_pad);
        }

        int block_size = 1;
        if (interp_kernel) {
            if (mayiuse(avx512_common)) {
                block_size = 16;
            } else {
                block_size = 8;
            }
        }

        // Align channel number to block size to deal with channels padding in IE with multiple blobs
        size_t CB = (C + block_size - 1) & (-block_size);

        size_t CH = (C + block_size - 1) / block_size;

        parallel_for3d(N, CH, OH_pad, [&](size_t n, size_t cb, size_t h) {
                    const float *psrc_n_cb = src + n * CB * IH * IW + cb * block_size * IW * IH;  //  n+cb src address

                    // h is output h
                    float fh = rh * h;
                    // ih0 is higher input h position
                    int ih0 = static_cast<int>(fh);
                    // ih1 is lower input h position
                    int ih1 = (ih0 < IH_pad - 1) ? ih0 + 1 : ih0;

                    float h_lambda0 = fh - ih0;  // for lower input h weight
                    float h_lambda1 = 1.0f - h_lambda0;  // for higher input h weight

                    const float *psrc_h0 = psrc_n_cb + (y1 + ih0) * IW * block_size + x1 * block_size;
                    const float *psrc_h1 = psrc_n_cb + (y1 + ih1) * IW * block_size + x1 * block_size;
                    float *pdst_h = dst + n * CB * OH * OW + cb * block_size * OW * OH + (y2 + h) * OW * block_size + x2 * block_size;

                    auto arg = jit_args_interp();
                    arg.h_lambda0 = static_cast<float*>(&h_lambda0);
                    arg.h_lambda1 = static_cast<float*>(&h_lambda1);
                    for (int w = 0; w < OW_pad; ++w) {
                        float fw = rw * w;
                        int iw0 = static_cast<int>(fw);
                        int iw1 = (iw0 < IW_pad - 1) ? iw0 + 1 : iw0;

                        float w_lambda0 = fw - iw0;  // for right input w weight
                        float w_lambda1 = 1.0f - w_lambda0;  // for left input w weight

                        const float *psrc00 = psrc_h0 + iw0 * block_size;
                        const float *psrc01 = psrc_h0 + iw1 * block_size;
                        const float *psrc10 = psrc_h1 + iw0 * block_size;
                        const float *psrc11 = psrc_h1 + iw1 * block_size;

                        float *pdst = pdst_h + w * block_size;

                        if (interp_kernel) {
                            arg.src00 = psrc00;
                            arg.src01 = psrc01;
                            arg.src10 = psrc10;
                            arg.src11 = psrc11;
                            arg.dst = pdst;
                            arg.w_lambda0 = static_cast<float*>(&w_lambda0);
                            arg.w_lambda1 = static_cast<float*>(&w_lambda1);
                            (*interp_kernel)(&arg);
                        } else {
                            for (int c = 0; c < block_size; ++c) {
                                pdst[c] = h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                                    h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
                            }
                        }
                    }
        });
    }

    void interpolate_8u(Layout layout, const size_t N, const size_t C,
        const uint8_t *src, const int x1, const int y1,
        const int IH_pad, const int IW_pad, const size_t IH, const size_t IW,
        float *dst, const int x2, const int y2,
        const int OH_pad, const int OW_pad, const size_t OH, const size_t OW) {
        if (IH_pad == OH_pad && IW_pad == OW_pad) {
            for (size_t i = 0; i < N * C * OH * OW; i++) {
                dst[i] = static_cast<float>(src[i]);
            }
            return;
        }

        float rh;
        float rw;
        if (align_corners) {
            rh = (OH_pad > 1) ? static_cast<float>(IH_pad - 1) / (OH_pad - 1) : 0.0f;
            rw = (OW_pad > 1) ? static_cast<float>(IW_pad - 1) / (OW_pad - 1) : 0.0f;
        } else {
            rh = static_cast<float>(IH_pad) / (OH_pad);
            rw = static_cast<float>(IW_pad) / (OW_pad);
        }

        parallel_for3d(N, C, OH_pad, [&](size_t n, size_t cb, size_t h) {
            const uint8_t *psrc = src + n * C * IH * IW;

            float fh = rh * h;
            int ih0 = static_cast<int>(fh);
            int ih1 = (ih0 < IH_pad - 1) ? ih0 + 1 : ih0;

            float h_lambda0 = fh - ih0;
            float h_lambda1 = 1.0f - h_lambda0;

            for (int w = 0; w < OW_pad; ++w) {
                float fw = rw * w;
                int iw0 = static_cast<int>(fw);
                int iw1 = (iw0 < IW_pad - 1) ? iw0 + 1 : iw0;

                float w_lambda0 = fw - iw0;
                float w_lambda1 = 1.0f - w_lambda0;

                dst[n * C * OH * OW + cb * OW * OH + (y2 + h) * OW + (x2 + w)] =
                    h_lambda1 * (w_lambda1 * static_cast<float>(psrc[cb * IW * IH + (y1 + ih0) * IW + (x1 + iw0)]) +
                    w_lambda0 * static_cast<float>(psrc[cb * IW * IH + (y1 + ih0) * IW + (x1 + iw1)])) +
                    h_lambda0 * (w_lambda1 * static_cast<float>(psrc[cb * IW * IH + (y1 + ih1) * IW + (x1 + iw0)]) +
                    w_lambda0 * static_cast<float>(psrc[cb * IW * IH + (y1 + ih1) * IW + (x1 + iw1)]));
            }
        });
    }
};

REG_FACTORY_FOR(InterpImpl, Interp);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
