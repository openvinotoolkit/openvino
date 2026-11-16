#pragma once
#include <CL/cl.h>

#include "../iMegakernelRuntime.h"

namespace mk {

// Which subgraph a MegaKernel op replaces. Carried in MegaKernelAttrs::params
// (see Gemma4KindMeta below), since the generic MegaKernelAttrs no longer has
// a model-specific kind field.
enum Gemma4Kind : int {
    kFfn = 1,     // post-attention residual .. layer_scalar (dense MLP + MoE)
    kAttnIn = 2,  // input_layernorm .. q/k/v projections, norms and RoPE
    kMaskFull = 3,     // attention mask of the full-attention layers
    kMaskSliding = 4,  // attention mask of the sliding-window layers
    kModel = 5,        // the whole language model: embeddings in, logits out
};

// Packed into every Gemma4 MegaKernel's MegaKernelAttrs::params (first field
// always `kind`, so mk_kind() in the transformation can read it generically).
struct Gemma4KindMeta {
    int32_t kind;
    int32_t num_layers = 0;  // only meaningful for kModel
};

// Shapes of the Gemma-4-26B-A4B FFN block; the transformation refuses layers
// that do not match.
constexpr int kHidden = 2816;
constexpr int kInter = 2112;
constexpr int kExpertInter = 704;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kQuantGroup = 64;

// Graph port order of a kFfn MegaKernel, fixed by gemma4POCOVIntegration.cpp.
// Int4 matrices arrive repacked into sub-group records (repack_row_blocks).
enum FfnPort : int {
    kFfnO = 0,          // f16 [B,S,KA] attention output (o_proj input)
    kFfnH,              // f16 [B,S,H]  residual entering the layer
    kFfnWPostAttn,      // f16 [H]      post_attention_layernorm
    kFfnWPreFf,         // f16 [H]      pre_feedforward_layernorm
    kFfnGate,           // u8 records   mlp.gate_proj [I,H]
    kFfnUp,             // u8 records   mlp.up_proj   [I,H]
    kFfnDown,           // u8 records   mlp.down_proj [H,I]
    kFfnWPostFf1,       // f16 [H]      post_feedforward_layernorm_1
    kFfnWRouter,        // f16 [H]      gamma of the RMS feeding router.proj
    kFfnRouterW,        // f16 [E,H]
    kFfnWPreFf2,        // f16 [H]      pre_feedforward_layernorm_2
    kFfnEGate,          // u8 records   experts gate [E*IE,H]
    kFfnEUp,            // u8 records   experts up   [E*IE,H]
    kFfnEDown,          // u8 records   experts down [E*H,IE]
    kFfnWPostFf2,       // f16 [H]      post_feedforward_layernorm_2
    kFfnWPostFf,        // f16 [H]      post_feedforward_layernorm
    kFfnLayerScalar,    // f16 [1]
    kFfnOProj,          // u8 records   self_attn.o_proj [H,KA]
    kFfnPorts
};

// Graph port order of a kAttnIn MegaKernel. Its output is f16 [B,S,(HQ+2*HK)*D]
// holding the roped q, the roped k and the normalised v of each token.
enum AttnPort : int {
    kAttnX = 0,         // f16 [B,S,H]   residual entering the layer
    kAttnPos,           // i32/i64 [B,S] position_ids
    kAttnWIn,           // f16 [H]       input_layernorm
    kAttnQkv,           // u8 records    q|k|v projection rows [R,H]
    kAttnWQ,            // f16 [D]       q_norm
    kAttnWK,            // f16 [D]       k_norm
    kAttnMeta,          // i32 [16]      see attn_post in kFfnSource
    kAttnInvFreq,       // f32 [C/2]     rotary_emb inv_freq
    kAttnPorts
};
constexpr int kAttnMaxHeads = 64;
constexpr int kAttnMaxRows = 16384;

// Graph port order of a kMaskFull / kMaskSliding MegaKernel; output [B,1,S,L].
enum MaskPort : int {
    kMaskAm = 0,        // i32/i64 [B,L] attention_mask
    kMaskTti,           // i32/i64 [B,S] token_type_ids
    kMaskMeta,          // f32 [16]      see mask_fill in kFfnSource
    kMaskPorts
};

// Graph port order of a kModel MegaKernel; output f32 logits [B, S, V] (or
// [B, 1, V] when the graph only asks for the last token). Layer l's weights
// follow at kModelLayer0 + l * kModelLayerPorts: the kAttnWIn.. ports of its
// kAttnIn MegaKernel, then the kFfnWPostAttn.. ports of its kFfn MegaKernel.
enum ModelPort : int {
    kModelEmbeds = 0,   // f32 [B,S,H]   inputs_embeds
    kModelPos,          // i32/i64 [B,S] position_ids
    kModelAm,           // i32/i64 [B,L] attention_mask
    kModelTti,          // i32/i64 [B,S] token_type_ids
    kModelMaskMeta,     // f32 [16]      mask_fill meta
    kModelMeta,         // f32 [kModelMetaLayer0 + layers], see ModelMeta
    kModelFinalNorm,    // f16 [H]
    kModelLmW,          // u8 lm_head records (repack_lm_head)
    kModelLmScale,      // f16 [V]
    kModelLmZp,         // u8  [V]
    kModelLayer0
};
constexpr int kModelAttnPorts = kAttnPorts - kAttnWIn;
constexpr int kModelFfnPorts = kFfnPorts - kFfnWPostAttn;
constexpr int kModelLayerPorts = kModelAttnPorts + kModelFfnPorts;
constexpr int kModelMaxLayers = 64;

// kModelMeta entries.
enum ModelMeta : int {
    kMetaLayers = 0,    // decoder layers
    kMetaVocab,         // V
    kMetaCapDiv,        // logits = tanh(y / cap_div) * cap_mul; cap_div 0 = no soft-cap
    kMetaCapMul,
    kMetaAttnScale,     // SDPA scale
    kMetaLastOnly,      // 1: logits of the last token only
    kMetaLayer0 = 16,   // per layer: 1 = sliding-window attention
};

class Gemma4ConstantParams : public IConstantParams {
public:
    int kind = kFfn;

    // kFfn, indexed by FfnPort; activation ports stay null.
    void* ffn[kFfnPorts] = {};

    // kAttnIn, indexed by AttnPort; activation ports stay null.
    void* attn[kAttnPorts] = {};
    int attn_rows = 0;  // rows of the kAttnQkv matrix

    void* mask_meta = nullptr;  // kMask*

    // kModel: device pointers of every port (activation ports stay null).
    int layers = 0;
    void* model[kModelLayer0] = {};
    void* layer[kModelMaxLayers][kModelLayerPorts] = {};
    int layer_qkv_rows[kModelMaxLayers] = {};
};

class Gemma4RuntimeParams : public IRuntimeParams {
public:
    void* in = nullptr;    // kLmHead: f16 [tokens, K]   kFfn: attention output [tokens, ka]   kAttnIn: x
    cl_mem in_mem = nullptr;  // kModel: inputs_embeds when it is a cl_mem buffer instead of USM (`in` is null)
    void* in2 = nullptr;   // kFfn: residual   kAttnIn: position_ids
    void* in3 = nullptr;   // kModel: attention_mask
    void* in4 = nullptr;   // kModel: token_type_ids
    void* out = nullptr;   // kLmHead: f32 [tokens, N]   kFfn: f16 [tokens, H]
    int tokens = 0;        // kMask*: batch
    int ka = 0;            // kFfn: o_proj input width   kAttnIn: RoPE table width   kMask*, kModel: L
    int s = 0;             // kMask*: query length
    int batch = 1;         // kModel
};

class Gemma4PlatformParams : public IPlatformParams {
public:
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue stream;
};

using ConstantParamsImpl = Gemma4ConstantParams;
using RuntimeParamsImpl = Gemma4RuntimeParams;
using PlatformParamsImpl = Gemma4PlatformParams;

}  // namespace mk
