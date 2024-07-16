# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from huggingface_hub import model_info
from models_hub_common.constants import hf_hub_cache_dir
from models_hub_common.utils import cleanup_dir
import transformers
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, AutoFeatureExtractor, AutoModelForTextToWaveform, \
    CLIPFeatureExtractor, XCLIPVisionModel, T5Tokenizer, VisionEncoderDecoderModel, ViTImageProcessor, BlipProcessor, BlipForConditionalGeneration, \
    SpeechT5Processor, SpeechT5ForTextToSpeech, LayoutLMv2Processor, Pix2StructForConditionalGeneration, RetriBertTokenizer, VivitImageProcessor

from torch_utils import TestTorchConvertModel, process_pytest_marks

def is_gptq_model(config):
    config_dict = config.to_dict() if not isinstance(config, dict) else config
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


def patch_gptq():
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    torch.set_default_dtype(torch.float32)
    torch.cuda.is_available = lambda: True

    from optimum.gptq import GPTQQuantizer

    orig_post_init_model = GPTQQuantizer.post_init_model

    def post_init_model(self, model):
        from auto_gptq import exllama_set_max_input_length

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = self.desc_act
        if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
            model = exllama_set_max_input_length(model, self.max_input_length)
        return model

    GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model


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


class TestTransformersModel(TestTorchConvertModel):
    def setup_class(self):
        from PIL import Image
        import requests

        self.infer_timeout = 1800

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)
        self.cuda_available, self.gptq_postinit = None, None

    def load_model(self, name, type):
        name_suffix = ''
        if name.find(':') != -1:
            name_suffix = name[name.find(':') + 1:]
            name = name[:name.find(':')]

        mi = model_info(name)
        auto_processor = None
        model = None
        example = None
        try:
            config = AutoConfig.from_pretrained(name)
        except Exception:
            config = {}
        is_gptq = is_gptq_model(config)
        model_kwargs = {"torchscript": True}
        if is_gptq:
            self.cuda_available, self.gptq_postinit = patch_gptq()
            model_kwargs["torch_dtype"] = torch.float32
            self.ov_config = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        if "bart" in mi.tags:
            model_kwargs["attn_implementation"] = "eager"
        try:
            auto_model = mi.transformersInfo['auto_model']
            if "processor" in mi.transformersInfo:
                auto_processor = mi.transformersInfo['processor']
        except:
            auto_model = None
        if "clip_vision_model" in mi.tags:
            preprocessor = CLIPFeatureExtractor.from_pretrained(name)
            encoded_input = preprocessor(self.image, return_tensors='pt')
            example = dict(encoded_input)
        elif 'xclip' in mi.tags:
            model = XCLIPVisionModel.from_pretrained(name, **model_kwargs)
            # needs video as input
            example = {'pixel_values': torch.randn(*(16, 3, 224, 224), dtype=torch.float32)}
        elif 'audio-spectrogram-transformer' in mi.tags:
            example = {'input_values': torch.randn(*(1, 1024, 128), dtype=torch.float32)}
        elif 'mega' in mi.tags:
            model = AutoModel.from_pretrained(name, **model_kwargs)
            model.config.output_attentions = True
            model.config.output_hidden_states = True
            model.config.return_dict = True
            example = dict(model.dummy_inputs)
        elif 'bros' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            encoding = processor("to the moon!", return_tensors="pt")
            bbox = torch.randn([1, 6, 8], dtype=torch.float32)
            example = dict(input_ids=encoding["input_ids"], bbox=bbox, attention_mask=encoding["attention_mask"])
        elif 'upernet' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            example = dict(processor(images=self.image, return_tensors="pt"))
        elif 'deformable_detr' in mi.tags or 'oneformer' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            example = dict(processor(images=self.image, task_inputs=["semantic"], return_tensors="pt"))
        elif 'clap' in mi.tags:
            example_inputs_map = {
                'audio_model': {'input_features': torch.randn([1, 1, 1001, 64], dtype=torch.float32)},
                'audio_projection': {'hidden_states': torch.randn([1, 768], dtype=torch.float32)},
            }
            example = example_inputs_map[name_suffix]
        elif 'git' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            example = {'pixel_values': torch.randn(*(1, 3, 224, 224), dtype=torch.float32), 
                       'input_ids': torch.randint(1, 100, size=(1, 13), dtype=torch.int64)}
        elif 'blip-2' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            example = dict(processor(images=self.image, return_tensors="pt"))
            example_inputs_map = {
                'vision_model' :  {'pixel_values': torch.randn([1, 3, 224, 224], dtype=torch.float32)},
                'qformer': {'query_embeds' : torch.randn([1, 32, 768], dtype=torch.float32), 
                            'encoder_hidden_states' : torch.randn([1, 257, 1408], dtype=torch.float32),
                            'encoder_attention_mask' : torch.ones([1, 257], dtype=torch.int64)},
                'language_projection': {'input' : torch.randn([1, 32, 768], dtype=torch.float32)},
            }
            example = example_inputs_map[name_suffix]
        elif "t5" in mi.tags:
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
            model = VisionEncoderDecoderModel.from_pretrained(
                name, **model_kwargs)
            feature_extractor = ViTImageProcessor.from_pretrained(name)
            encoded_input = feature_extractor(
                images=[self.image], return_tensors="pt")

            example = dict(encoded_input)
            example["decoder_input_ids"] = torch.randint(0, 1000, [1, 20])
            example["decoder_attention_mask"] = torch.ones([1, 20], dtype=torch.int64)
        elif 'idefics' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            
            prompts = [[
                "User: What is in this image?",
                "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                "<end_of_utterance>",

                "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

                "\nUser:",
                "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
                "And who is that?<end_of_utterance>",

                "\nAssistant:",
            ]]

            inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt")
            example = dict(inputs)
        elif 'blip' in mi.tags and 'text2text-generation' in mi.tags:
            processor = BlipProcessor.from_pretrained(name)
            model = BlipForConditionalGeneration.from_pretrained(name, **model_kwargs)
            text = "a photography of"
            inputs = processor(self.image, text, return_tensors="pt")
            example = dict(inputs)
        elif 'speecht5' in mi.tags:
            from datasets import load_dataset

            processor = SpeechT5Processor.from_pretrained(name)
            model = SpeechT5ForTextToSpeech.from_pretrained(name, **model_kwargs)

            inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
            # load xvector containing speaker's voice characteristics from a dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

            example = dict(inputs)
            example['speaker_embeddings'] = speaker_embeddings
            example['decoder_input_values'] = torch.randn([1, 20, model.config.num_mel_bins])
        elif 'layoutlmv2' in mi.tags:
            processor = LayoutLMv2Processor.from_pretrained(name)

            question = "What's the content of this image?"
            encoding = processor(self.image, question, max_length=512, truncation=True, return_tensors="pt")
            example = dict(encoding)
        elif 'pix2struct' in mi.tags:
            model = Pix2StructForConditionalGeneration.from_pretrained(name, **model_kwargs)
            processor = AutoProcessor.from_pretrained(name)

            import requests
            from PIL import Image
            image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
            image = Image.open(requests.get(image_url, stream=True).raw)
            question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
            inputs = processor(images=image, text=question, return_tensors="pt")
            example = dict(inputs)
            example["decoder_input_ids"] = torch.randint(0, 1000, [1, 20])
            example["decoder_attention_mask"] = torch.ones([1, 20], dtype=torch.int64)
        elif "mms-lid" in name:
            processor = AutoFeatureExtractor.from_pretrained(name)
            input_values = processor(torch.randn(16000).numpy(),
                                     sampling_rate=16_000,
                                     return_tensors="pt")
            example = {"input_values": input_values.input_values}
        elif "retribert" in mi.tags:
            tokenizer = RetriBertTokenizer.from_pretrained(name)
            encoding1 = tokenizer(
                "How many cats are there?", return_tensors="pt")
            encoding2 = tokenizer("Second text", return_tensors="pt")
            example = (encoding1.input_ids, encoding1.attention_mask,
                       encoding2.input_ids, encoding2.attention_mask)
        elif "mgp-str" in mi.tags or "clip_vision_model" in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(images=self.image, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "flava" in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(text=["a photo of a cat", "a photo of a dog"],
                                      images=[self.image, self.image],
                                      return_tensors="pt")
            example = dict(encoded_input)
        elif "vivit" in mi.tags:
            frames = list(torch.randint(
                0, 255, [32, 3, 224, 224]).to(torch.float32))
            processor = VivitImageProcessor.from_pretrained(name)
            encoded_input = processor(images=frames, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "tvlt" in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            num_frames = 8
            images = list(torch.rand(num_frames, 3, 224, 224))
            audio = list(torch.randn(10000))
            input_dict = processor(
                images, audio, sampling_rate=44100, return_tensors="pt")
            example = dict(input_dict)
        elif "gptsan-japanese" in mi.tags:
            processor = AutoTokenizer.from_pretrained(name)
            text = "織田信長は、"
            encoded_input = processor(text=[text], return_tensors="pt")
            example = dict(input_ids=encoded_input.input_ids,
                           token_type_ids=encoded_input.token_type_ids)
        elif "videomae" in mi.tags or "timesformer" in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            video = list(torch.randint(
                0, 255, [16, 3, 224, 224]).to(torch.float32))
            inputs = processor(video, return_tensors="pt")
            example = dict(inputs)
        elif 'text-to-speech' in mi.tags:
            tokenizer = AutoTokenizer.from_pretrained(name)
            text = "some example text in the English language"
            inputs = tokenizer(text, return_tensors="pt")
            example = dict(inputs)
        elif 'musicgen' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)
            model = AutoModelForTextToWaveform.from_pretrained(name, **model_kwargs)
            
            inputs = processor(
                text=["80s pop track with bassy drums and synth"],
                padding=True,
                return_tensors="pt",
            )
            example = dict(inputs)
            # works for facebook/musicgen-small
            pad_token_id = model.generation_config.pad_token_id
            example["decoder_input_ids"] = torch.ones(
                (inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long) * pad_token_id
        elif 'kosmos-2' in mi.tags:
            processor = AutoProcessor.from_pretrained(name)

            prompt = "<grounding>An image of"
            inputs = processor(text=prompt, images=self.image, return_tensors="pt")
            example = dict(inputs)
        else:
            try:
                if auto_model == "AutoModelForCausalLM":
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors='pt')
                    example = dict(encoded_input)
                    if "facebook/incoder" in name and "token_type_ids" in example:
                        del example["token_type_ids"]
                elif auto_model == "AutoModelForMaskedLM":
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors='pt')
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForImageClassification":
                    processor = AutoProcessor.from_pretrained(name)
                    encoded_input = processor(
                        images=self.image, return_tensors="pt")
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForSeq2SeqLM":
                    tokenizer = AutoTokenizer.from_pretrained(name)
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
                    from datasets import load_dataset
                    processor = AutoProcessor.from_pretrained(name)
                    inputs = processor(torch.randn(1000).numpy(),
                                       sampling_rate=16000,
                                       return_tensors="pt")
                    example = dict(inputs)
                elif auto_model == "AutoModelForCTC":
                    from datasets import load_dataset
                    processor = AutoProcessor.from_pretrained(name)
                    input_values = processor(torch.randn(1000).numpy(),
                                             return_tensors="pt")
                    example = dict(input_values)
                elif auto_model == "AutoModelForTableQuestionAnswering":
                    import pandas as pd
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
                            "Number of movies": ["87", "53", "69"]}
                    queries = ["What is the name of the first actor?",
                               "How many movies has George Clooney played in?",
                               "What is the total number of movies?", ]
                    answer_coordinates = [[(0, 0)], [(2, 1)], [
                        (0, 1), (1, 1), (2, 1)]]
                    answer_text = [["Brad Pitt"], ["69"], ["209"]]
                    table = pd.DataFrame.from_dict(data)
                    encoded_input = tokenizer(table=table, queries=queries, answer_coordinates=answer_coordinates,
                                              answer_text=answer_text, padding="max_length", return_tensors="pt", )
                    example = dict(input_ids=encoded_input["input_ids"],
                                   token_type_ids=encoded_input["token_type_ids"],
                                   attention_mask=encoded_input["attention_mask"])
                else:
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
            model = self.load_model_with_default_class(name, **model_kwargs)
        if hasattr(model, "set_default_language"):
            model.set_default_language("en_XX")
        if name_suffix != '':
            model = model._modules[name_suffix]
        if example is None:
            if "encodec" in mi.tags:
                example = (torch.randn(1, 1, 100),)
            else:
                example = (torch.randint(1, 1000, [1, 100]),)
        self.example = filter_example(model, example)
        if "vit_mae" in mi.tags:
            # vit-mae by default will generate random noise
            self.example["noise"] = torch.rand(1, 192)
        model.eval()
        # do first inference
        if isinstance(self.example, dict):
            model(**self.example)
        else:
            model(*self.example)
        return model

    def teardown_method(self):
        # remove all downloaded files from cache
        cleanup_dir(hf_hub_cache_dir)
        # restore after gptq patching
        if self.cuda_available is not None:
            unpatch_gptq(self.cuda_available, self.gptq_postinit)
            self.cuda_available, self.gptq_postinit = None, None
        super().teardown_method()

    @staticmethod
    def load_model_with_default_class(name, **kwargs):
        try:
            mi = model_info(name)
            assert len({"owlv2", "owlvit", "vit_mae"}.intersection(mi.tags)) == 0, "TBD: support default classes of these models"
            assert "architectures" in mi.config and len(mi.config["architectures"]) == 1
            class_name = mi.config["architectures"][0]
            model_class = transformers.__getattr__(class_name)
            return model_class.from_pretrained(name, **kwargs)
        except:
            return AutoModel.from_pretrained(name, **kwargs)

    @pytest.mark.parametrize("name,type", [("allenai/led-base-16384", "led"),
                                           ("bert-base-uncased", "bert"),
                                           ("google/flan-t5-base", "t5"),
                                           ("google/tapas-large-finetuned-wtq", "tapas"),
                                           ("gpt2", "gpt2"),
                                           ("openai/clip-vit-large-patch14", "clip"),
                                           ("katuni4ka/opt-125m-gptq", "opt"),
                                           ])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)

    @pytest.mark.parametrize("name",
                             process_pytest_marks(os.path.join(os.path.dirname(__file__), "hf_transformers_models")))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, ie_device):
        self.run(model_name=name, model_link=None, ie_device=ie_device)
