# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from huggingface_hub import model_info
from models_hub_common.test_convert_model import TestConvertModel
from openvino import convert_model
from models_hub_common.utils import get_models_list, cleanup_dir
from models_hub_common.constants import hf_hub_cache_dir


def flattenize_tuples(list_input):
    unpacked_pt_res = []
    for r in list_input:
        if isinstance(r, (tuple, list)):
            unpacked_pt_res.extend(flattenize_tuples(r))
        else:
            unpacked_pt_res.append(r)
    return unpacked_pt_res


def flattenize_outputs(outputs):
    if not isinstance(outputs, dict):
        outputs = flattenize_tuples(outputs)
        return [i.numpy(force=True) for i in outputs]
    else:
        return dict((k, v.numpy(force=True)) for k, v in outputs.items())


def filter_example(model, example):
    try:
        import inspect
        if isinstance(example, dict):
            model_params = inspect.signature(model.forward).parameters
            names_set = {p for p in model_params}
            new_example = dict()
            for k, v in example:
                if k in names_set:
                    new_example[k] = v
        return new_example
    except:
        return example


# To make tests reproducible we seed the random generator
torch.manual_seed(0)

visual_question_answer_models = [
    "google/pix2struct-docvqa-base",
    'google/pix2struct-ai2d-base',
]

class TestTransformersModel(TestConvertModel):
    def setup_class(self):
        from PIL import Image
        import requests

        self.infer_timeout = 800

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def load_model(self, name, type):
        mi = model_info(name)
        auto_processor = None
        model = None
        example = None
        try:
            auto_model = mi.transformersInfo['auto_model']
            if "processor" in mi.transformersInfo:
                auto_processor = mi.transformersInfo['processor']
        except:
            auto_model = None
        if "clip_vision_model" in mi.tags:
            from transformers import CLIPVisionModel, CLIPFeatureExtractor
            model = CLIPVisionModel.from_pretrained(name, torchscript=True)
            preprocessor = CLIPFeatureExtractor.from_pretrained(name)
            encoded_input = preprocessor(self.image, return_tensors='pt')
            example = dict(encoded_input)
        elif "t5" in mi.tags:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(name)
            encoder = tokenizer(
                "Studies have been shown that owning a dog is good for you", return_tensors="pt")
            decoder = tokenizer("Studies show that", return_tensors="pt")
            example = (encoder.input_ids, encoder.attention_mask,
                       decoder.input_ids, decoder.attention_mask)
        elif "hubert" in mi.tags:
            wav_input_16khz = torch.randn(1, 10000)
            example = (wav_input_16khz,)
        elif "vit-gpt2" in name:
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor
            model = VisionEncoderDecoderModel.from_pretrained(
                name, torchscript=True)
            feature_extractor = ViTImageProcessor.from_pretrained(name)
            encoded_input = feature_extractor(
                images=[self.image], return_tensors="pt")

            class VIT_GPT2_Model(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model.generate(x, max_length=16, num_beams=4)

            model = VIT_GPT2_Model(model)
            example = (encoded_input.pixel_values,)
        elif "mms-lid" in name:
            # mms-lid model config does not have auto_model attribute, only direct loading aviable 
            from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
            model = Wav2Vec2ForSequenceClassification.from_pretrained(name, torchscript=True)
            processor = AutoFeatureExtractor.from_pretrained(name)
            input_values = processor(torch.randn(16000).numpy(), sampling_rate=16_000, return_tensors="pt")
            example =  {"input_values": input_values.input_values}
        elif "retribert" in mi.tags:
            from transformers import RetriBertTokenizer
            text = "How many cats are there?"
            tokenizer = RetriBertTokenizer.from_pretrained(name)
            encoding1 = tokenizer(
                "How many cats are there?", return_tensors="pt")
            encoding2 = tokenizer("Second text", return_tensors="pt")
            example = (encoding1.input_ids, encoding1.attention_mask,
                       encoding2.input_ids, encoding2.attention_mask)
        elif "mgp-str" in mi.tags or "clip_vision_model" in mi.tags:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(images=self.image, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "flava" in mi.tags:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(text=["a photo of a cat", "a photo of a dog"],
                                      images=[self.image, self.image],
                                      return_tensors="pt")
            example = dict(encoded_input)
        elif "vivit" in mi.tags:
            from transformers import VivitImageProcessor
            frames = list(torch.randint(
                0, 255, [32, 3, 224, 224]).to(torch.float32))
            processor = VivitImageProcessor.from_pretrained(name)
            encoded_input = processor(images=frames, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "tvlt" in mi.tags:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(name)
            num_frames = 8
            images = list(torch.rand(num_frames, 3, 224, 224))
            audio = list(torch.randn(10000))
            input_dict = processor(
                images, audio, sampling_rate=44100, return_tensors="pt")
            example = dict(input_dict)
        elif "gptsan-japanese" in mi.tags:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(name)
            text = "織田信長は、"
            encoded_input = processor(text=[text], return_tensors="pt")
            example = dict(input_ids=encoded_input.input_ids,
                           token_type_ids=encoded_input.token_type_ids)
        elif "videomae" in mi.tags or "timesformer" in mi.tags:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(name)
            video = list(torch.randint(
                0, 255, [16, 3, 224, 224]).to(torch.float32))
            inputs = processor(video, return_tensors="pt")
            example = dict(inputs)
        else:
            try:
                if auto_model == "AutoModelForCausalLM":
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForCausalLM.from_pretrained(
                        name, torchscript=True)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors='pt')
                    inputs_dict = dict(encoded_input)
                    if "facebook/incoder" in name and "token_type_ids" in inputs_dict:
                        del inputs_dict["token_type_ids"]
                    example = inputs_dict
                elif auto_model == "AutoModelForMaskedLM":
                    from transformers import AutoTokenizer, AutoModelForMaskedLM
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForMaskedLM.from_pretrained(
                        name, torchscript=True)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors='pt')
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForImageClassification":
                    from transformers import AutoProcessor, AutoModelForImageClassification
                    processor = AutoProcessor.from_pretrained(name)
                    model = AutoModelForImageClassification.from_pretrained(
                        name, torchscript=True)
                    encoded_input = processor(
                        images=self.image, return_tensors="pt")
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForSeq2SeqLM" and name not in visual_question_answer_models:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        name, torchscript=True)
                    inputs = tokenizer(
                        "Studies have been shown that owning a dog is good for you", return_tensors="pt")
                    decoder_inputs = tokenizer(
                        "<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    example = dict(input_ids=inputs.input_ids,
                                   decoder_input_ids=decoder_inputs.input_ids)
                elif auto_model == "AutoModelForSpeechSeq2Seq":
                    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                    from datasets import load_dataset
                    processor = AutoProcessor.from_pretrained(name)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        name, torchscript=True)
                    inputs = processor(torch.randn(1000).numpy(), sampling_rate=16000, return_tensors="pt")
                    example = dict(inputs)
                elif auto_model == "AutoModelForCTC":
                    from transformers import AutoProcessor, AutoModelForCTC
                    from datasets import load_dataset
                    processor = AutoProcessor.from_pretrained(name)
                    model = AutoModelForCTC.from_pretrained(
                        name, torchscript=True)
                    input_values = processor(torch.randn(1000).numpy(), return_tensors="pt")
                    example = dict(input_values)
                elif auto_model == "AutoModelForTableQuestionAnswering":
                    import pandas as pd
                    from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForTableQuestionAnswering.from_pretrained(
                        name, torchscript=True)
                    data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
                            "Number of movies": ["87", "53", "69"]}
                    queries = ["What is the name of the first actor?",
                               "How many movies has George Clooney played in?",
                               "What is the total number of movies?",]
                    answer_coordinates = [[(0, 0)], [(2, 1)], [
                        (0, 1), (1, 1), (2, 1)]]
                    answer_text = [["Brad Pitt"], ["69"], ["209"]]
                    table = pd.DataFrame.from_dict(data)
                    encoded_input = tokenizer(table=table, queries=queries, answer_coordinates=answer_coordinates,
                                              answer_text=answer_text, padding="max_length", return_tensors="pt",)
                    example = dict(input_ids=encoded_input["input_ids"],
                                   token_type_ids=encoded_input["token_type_ids"],
                                   attention_mask=encoded_input["attention_mask"])
                elif auto_model == 'AutoModelForTextToWaveform':
                    from transformers import AutoProcessor, MusicgenForConditionalGeneration
                    processor = AutoProcessor.from_pretrained(name)
                    model = MusicgenForConditionalGeneration.from_pretrained(name, torchscript=True)

                    class MusicGenerateModel(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, **x):
                            return self.model.generate(**x, max_new_tokens=256)
                    
                    inputs = processor(
                        text=["80s pop track with bassy drums and synth"],
                        padding=True,
                        return_tensors="pt",
                    )
                    example = dict(inputs)        
                    model = MusicGenerateModel(model)
                elif auto_model == 'RagTokenForGeneration':
                    from transformers import AutoTokenizer, RagTokenForGeneration#, RagRetriever
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    # retriever = RagRetriever.from_pretrained(name, index_name="exact", use_dummy_dataset=True)
                    model = RagTokenForGeneration.from_pretrained(name)#, retriever=retriever)
                    
                    class RagTokenGenerationModel(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, x):
                            return self.model.generate(x)
                    
                    input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", 
                                            return_tensors="pt") 
                    example = input_dict['input_ids']

                    model = RagTokenGenerationModel(model)
                elif auto_model == 'AutoModelForSeq2SeqLM' and name in visual_question_answer_models:
                    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
                    model = Pix2StructForConditionalGeneration.from_pretrained(name)
                    processor = Pix2StructProcessor.from_pretrained(name)

                    import requests
                    from PIL import Image
                    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
                    image = Image.open(requests.get(image_url, stream=True).raw)
                    question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
                    inputs = processor(images=image, text=question, return_tensors="pt")
                    
                    class DecoratorModelForSeq2SeqLM(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, **x):
                            return self.model.generate(**x)
                    example = dict(inputs)
                    model = DecoratorModelForSeq2SeqLM(model)
                elif auto_model == 'RealmForOpenQA':
                    from transformers import AutoTokenizer, RealmForOpenQA
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = RealmForOpenQA.from_pretrained(name)
                    # todo: provide example input for google/realm-orqa-nq-openqa
                elif auto_model == 'AutoModelForTextToSpectrogram':
                    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                    from datasets import load_dataset
                    import torch

                    processor = SpeechT5Processor.from_pretrained(name)
                    model = SpeechT5ForTextToSpeech.from_pretrained(name)
                    vocoder = SpeechT5HifiGan.from_pretrained(name)

                    inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
                    # load xvector containing speaker's voice characteristics from a dataset
                    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                    example = {'input_ids': inputs["input_ids"], 'speaker_embeddings': speaker_embeddings, 'vocoder': vocoder}

                    class DecoratorModelForSeq2SeqLM(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, **x):
                            return self.model.generate_speech(**x)
                    model = DecoratorModelForSeq2SeqLM(model)
                else:
                    from transformers import AutoTokenizer, AutoProcessor
                    text = "Replace me by any text you'd like."
                    if auto_processor is not None and "Tokenizer" not in auto_processor:
                        processor = AutoProcessor.from_pretrained(name)
                        encoded_input = processor(
                            text=[text], images=self.image, return_tensors="pt", padding=True)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(name)
                        encoded_input = tokenizer(text, return_tensors='pt')
                    example = dict(encoded_input)
            except:
                pass
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(name, torchscript=True)
        if hasattr(model, "set_default_language"):
            model.set_default_language("en_XX")
        if example is None:
            if "encodec" in mi.tags:
                example = (torch.randn(1, 1, 100),)
            else:
                example = (torch.randint(1, 1000, [1, 100]),)
        self.example = filter_example(model, example)
        model.eval()
        # do first inference
        if isinstance(self.example, dict):
            model(**self.example)
        else:
            model(*self.example)
        return model

    def get_inputs_info(self, model_obj):
        return None

    def prepare_inputs(self, inputs_info):
        if isinstance(self.example, dict):
            return dict((k, v.numpy()) for k, v in self.example.items())
        else:
            return [i.numpy() for i in self.example]

    def convert_model(self, model_obj):
        ov_model = convert_model(model_obj,
                                 example_input=self.example,
                                 verbose=True)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        if isinstance(inputs, dict):
            inps = dict((k, torch.from_numpy(v)) for k, v in inputs.items())
            fw_outputs = model_obj(**inps)
        else:
            fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        return flattenize_outputs(fw_outputs)

    def teardown_method(self):
        # remove all downloaded files from cache
        cleanup_dir(hf_hub_cache_dir)
        super().teardown_method()

    @pytest.mark.parametrize("name,type", [("allenai/led-base-16384", "led"),
                                           ("bert-base-uncased", "bert"),
                                           ("google/flan-t5-base", "t5"),
                                           ("google/tapas-large-finetuned-wtq", "tapas"),
                                           ("gpt2", "gpt2"),
                                           ("openai/clip-vit-large-patch14", "clip")
                                           ])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)

    @pytest.mark.parametrize("name",
                             [pytest.param(n, marks=pytest.mark.xfail(reason=r) if m == "xfail" else pytest.mark.skip(reason=r)) if m else n for n, _, m, r in get_models_list(os.path.join(os.path.dirname(__file__), "hf_transformers_models"))])
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, ie_device):
        self.run(model_name=name, model_link=None, ie_device=ie_device)
