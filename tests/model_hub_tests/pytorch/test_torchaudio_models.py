# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torchaudio.models as tam
import torchaudio.pipelines as P
from models_hub_common.utils import get_models_list

from torch_utils import TestTorchConvertModel


# Mapping: torchaudio bundle name -> (factory, example_inputs, actual_inputs)
# example_inputs use one length, actual_inputs another - to exercise dynamic
# shape behaviour. The export mode also constrains the dynamic dimension via
# self.dynamo_input.

def _w2v_inputs():
    # Wav2Vec2 family takes (waveforms: [batch, time]).
    # Use batch=2 in the example so torch.export does not specialize batch to 1
    # and infer at batch=3 with a different audio length.
    example = (torch.randn(2, 16000),)
    inputs = (torch.randn(3, 32000),)
    return example, inputs


def _convtasnet_inputs():
    # 2s vs 4s mixture @16kHz, single channel
    example = (torch.randn(1, 1, 32000),)
    inputs = (torch.randn(1, 1, 64000),)
    return example, inputs


def _hdemucs_inputs():
    # 1s vs 2s stereo @44.1kHz
    example = (torch.randn(1, 2, 44100),)
    inputs = (torch.randn(1, 2, 88200),)
    return example, inputs


def _squim_objective_inputs():
    example = (torch.randn(1, 16000),)
    inputs = (torch.randn(1, 32000),)
    return example, inputs


def _squim_subjective_inputs():
    example = (torch.randn(1, 16000), torch.randn(1, 16000))
    inputs = (torch.randn(1, 32000), torch.randn(1, 32000))
    return example, inputs


def _rnnt_inputs():
    # (sources, source_lengths) with 200 vs 300 frames of 80-bin features
    example = (torch.randn(1, 200, 80), torch.tensor([200], dtype=torch.int32))
    inputs = (torch.randn(1, 300, 80), torch.tensor([300], dtype=torch.int32))
    return example, inputs


def _tacotron2_inputs():
    example = (torch.zeros(1, 30, dtype=torch.int64), torch.tensor([30], dtype=torch.int64))
    inputs = (torch.zeros(1, 50, dtype=torch.int64), torch.tensor([50], dtype=torch.int64))
    return example, inputs


def _wavernn_inputs():
    example = (torch.randn(1, 80, 100), torch.tensor([100], dtype=torch.int64))
    inputs = (torch.randn(1, 80, 200), torch.tensor([200], dtype=torch.int64))
    return example, inputs


class _Wav2Vec2Wrap(torch.nn.Module):
    # Wav2Vec2Model.forward returns (features, Optional[lengths]); lengths is
    # None when no input lengths are provided. Drop it so the model is
    # traceable and produces a single, comparable tensor output.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveforms):
        return self.model(waveforms)[0]


class _EmformerTranscriberWrap(torch.nn.Module):
    def __init__(self, rnnt):
        super().__init__()
        self.rnnt = rnnt

    def forward(self, sources, source_lengths):
        return self.rnnt.transcribe(sources, source_lengths)


class _Tacotron2InferWrap(torch.nn.Module):
    def __init__(self, t2):
        super().__init__()
        self.t2 = t2

    def forward(self, tokens, lengths):
        spec, spec_lens, _ = self.t2.infer(tokens, lengths)
        return spec, spec_lens


# bundle name -> (model factory, inputs factory, dynamic_dims_per_input)
# dynamic_dims_per_input[i] is a list of dim indices that should be marked
# dynamic (-1 in the input PartialShape) for input i in export mode. Empty
# means fully static export.
_W2V_DYN_DIMS = [[0, 1]]  # [batch, time]

_TORCHAUDIO_REGISTRY = {
    "WAV2VEC2_BASE":            (lambda: _Wav2Vec2Wrap(P.WAV2VEC2_BASE.get_model()),            _w2v_inputs,         _W2V_DYN_DIMS),
    "WAV2VEC2_ASR_BASE_960H":   (lambda: _Wav2Vec2Wrap(P.WAV2VEC2_ASR_BASE_960H.get_model()),   _w2v_inputs,         _W2V_DYN_DIMS),
    "HUBERT_BASE":              (lambda: _Wav2Vec2Wrap(P.HUBERT_BASE.get_model()),              _w2v_inputs,         _W2V_DYN_DIMS),
    "WAVLM_BASE":               (lambda: _Wav2Vec2Wrap(P.WAVLM_BASE.get_model()),               _w2v_inputs,         _W2V_DYN_DIMS),
    "MMS_FA":                   (lambda: _Wav2Vec2Wrap(P.MMS_FA.get_model()),                   _w2v_inputs,         _W2V_DYN_DIMS),
    "CONVTASNET_BASE_LIBRI2MIX":(lambda: P.CONVTASNET_BASE_LIBRI2MIX.get_model(),_convtasnet_inputs,  None),
    "HDEMUCS_HIGH_MUSDB":       (lambda: P.HDEMUCS_HIGH_MUSDB.get_model(),       _hdemucs_inputs,     None),
    "SQUIM_OBJECTIVE":          (lambda: P.SQUIM_OBJECTIVE.get_model(),          _squim_objective_inputs, None),
    "SQUIM_SUBJECTIVE":         (lambda: P.SQUIM_SUBJECTIVE.get_model(),         _squim_subjective_inputs, None),
    "EMFORMER_RNNT":            (lambda: _EmformerTranscriberWrap(tam.emformer_rnnt_base(num_symbols=4096)),
                                                                                  _rnnt_inputs,        None),
    "TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH":
        (lambda: _Tacotron2InferWrap(P.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_tacotron2()),
         _tacotron2_inputs, None),
    "TACOTRON2_WAVERNN_CHAR_LJSPEECH":
        (lambda: P.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_vocoder(),
         _wavernn_inputs, None),
}


# Reproducibility
torch.manual_seed(0)


class TestTorchaudioConvertModel(TestTorchConvertModel):
    def load_model(self, model_name, model_link):
        if model_name not in _TORCHAUDIO_REGISTRY:
            raise RuntimeError(f"Unknown torchaudio model: {model_name}")
        factory, inputs_fn, dyn_dims = _TORCHAUDIO_REGISTRY[model_name]
        model = factory()
        model.eval()
        example, inputs = inputs_fn()
        self.example = example
        if getattr(self, "mode", None) == "export" and dyn_dims is not None:
            # Build a PartialShape per input tensor with the listed dims
            # marked dynamic (-1). torch.export will treat those as Dim.AUTO,
            # and the resulting OV model accepts varying shapes at inference.
            from openvino import PartialShape
            specs = []
            for i, t in enumerate(example):
                if not isinstance(t, torch.Tensor):
                    specs.append(None)
                    continue
                shape = list(t.shape)
                dyn = dyn_dims[i] if i < len(dyn_dims) else []
                for d in dyn:
                    if 0 <= d < len(shape):
                        shape[d] = -1
                specs.append(PartialShape(shape))
            self.dynamo_input = tuple(specs)
            # Feed shapes different from the example to exercise the dynamic
            # axes end-to-end (conversion + compile + infer).
            self.inputs = inputs
        elif getattr(self, "mode", None) == "export":
            # Audio models have intricate shape divisibility constraints
            # (conv strides, attention block sizes, etc.) that make bounded
            # dynamic export fragile - keep static export for these by
            # reusing the example shape at inference.
            self.inputs = example
        else:
            # Trace produces a dynamic TorchScript graph natively; feed
            # a different shape at inference to exercise the dynamic path.
            self.inputs = inputs
        return model

    @pytest.mark.parametrize("model_name", ["WAV2VEC2_BASE"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, ie_device):
        self.mode = "trace"
        self.run(model_name, None, ie_device)

    @pytest.mark.parametrize("model_name", ["WAV2VEC2_BASE"])
    @pytest.mark.precommit
    def test_convert_model_precommit_export(self, model_name, ie_device):
        self.mode = "export"
        self.run(model_name, None, ie_device)

    @pytest.mark.parametrize("name,link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "torchaudio_models")))
    @pytest.mark.parametrize("mode", ["trace", "export"])
    @pytest.mark.nightly
    def test_convert_model_all_models(self, mode, name, link, mark, reason, ie_device):
        self.mode = mode
        assert mark is None or mark in [
            'skip', 'xfail', 'xfail_trace', 'xfail_export'], f"Incorrect test case for {name}"
        if mark == 'skip':
            pytest.skip(reason)
        elif mark in ['xfail', f'xfail_{mode}']:
            pytest.xfail(reason)
        self.run(name, None, ie_device)
