GenAI Use Cases
=====================

This article provides several use case scenarios for Generative AI model
inference. The applications presented in the code samples below
only require minimal configuration, like setting an inference device. Feel free
to explore and modify the source code as you need.


Using GenAI for Text-to-Image Generation
########################################

Examples below demonstrate inference on text-to-image models, like Stable Diffusion
1.5, 2.1, and LCM, with a text prompt as input. The :ref:`main.cpp <maincpp>`
sample shows basic usage of the ``Text2ImagePipeline`` pipeline.
:ref:`lora.cpp <loracpp>` shows how to apply LoRA adapters to the pipeline.


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. tab-set::

         .. tab-item:: main.py
            :name: mainpy

            .. code-block:: python

               import openvino_genai
               from PIL import Image
               import numpy as np

               class Generator(openvino_genai.Generator):
                   def __init__(self, seed, mu=0.0, sigma=1.0):
                       openvino_genai.Generator.__init__(self)
                       np.random.seed(seed)
                       self.mu = mu
                       self.sigma = sigma

                   def next(self):
                       return np.random.normal(self.mu, self.sigma)


               def infer(model_dir: str, prompt: str):
                   device = 'CPU'  # GPU can be used as well
                   random_generator = Generator(42)
                   pipe = openvino_genai.Text2ImagePipeline(model_dir, device)
                   image_tensor = pipe.generate(
                       prompt,
                       width=512,
                       height=512,
                       num_inference_steps=20,
                       num_images_per_prompt=1,
                       random_generator=random_generator
                   )

                   image = Image.fromarray(image_tensor.data[0])
                   image.save("image.bmp")

         .. tab-item:: LoRA.py
            :name: lorapy

            .. code-block:: python

               import openvino as ov
               import openvino_genai
               import numpy as np
               import sys


               class Generator(openvino_genai.Generator):
                   def __init__(self, seed, mu=0.0, sigma=1.0):
                       openvino_genai.Generator.__init__(self)
                       np.random.seed(seed)
                       self.mu = mu
                       self.sigma = sigma

                   def next(self):
                       return np.random.normal(self.mu, self.sigma)


               def image_write(path: str, image_tensor: ov.Tensor):
                   from PIL import Image
                   image = Image.fromarray(image_tensor.data[0])
                   image.save(path)


               def infer(models_path: str, prompt: str):
                   prompt = "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"

                   device = "CPU"  # GPU, NPU can be used as well
                   adapter_config = openvino_genai.AdapterConfig()

                   for i in range(int(len(adapters) / 2)):
                       adapter = openvino_genai.Adapter(adapters[2 * i])
                       alpha = float(adapters[2 * i + 1])
                       adapter_config.add(adapter, alpha)

                   pipe = openvino_genai.Text2ImagePipeline(models_path, device, adapters=adapter_config)
                   print("Generating image with LoRA adapters applied, resulting image will be in lora.bmp")
                   image = pipe.generate(prompt,
                                         random_generator=Generator(42),
                                         width=512,
                                         height=896,
                                         num_inference_steps=20)

                   image_write("lora.bmp", image)
                   print("Generating image without LoRA adapters applied, resulting image will be in baseline.bmp")
                   image = pipe.generate(prompt,
                                         adapters=openvino_genai.AdapterConfig(),
                                         random_generator=Generator(42),
                                         width=512,
                                         height=896,
                                         num_inference_steps=20
                                         )
                   image_write("baseline.bmp", image)

      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text2image/README.md>`__

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: main.cpp
            :name: maincpp

            .. code-block:: cpp

               #include "openvino/genai/text2image/pipeline.hpp"

               #include "imwrite.hpp"

               int32_t main(int32_t argc, char* argv[]) try {
                   OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

                   const std::string models_path = argv[1], prompt = argv[2];
                   const std::string device = "CPU";  // GPU, NPU can be used as well

                   ov::genai::Text2ImagePipeline pipe(models_path, device);
                   ov::Tensor image = pipe.generate(prompt,
                       ov::genai::width(512),
                       ov::genai::height(512),
                       ov::genai::num_inference_steps(20),
                       ov::genai::num_images_per_prompt(1));

                   imwrite("image_%d.bmp", image, true);

                   return EXIT_SUCCESS;
               } catch (const std::exception& error) {
                   try {
                       std::cerr << error.what() << '\n';
                   } catch (const std::ios_base::failure&) {}
                   return EXIT_FAILURE;
               } catch (...) {
                   try {
                       std::cerr << "Non-exception object thrown\n";
                   } catch (const std::ios_base::failure&) {}
                   return EXIT_FAILURE;
               }

         .. tab-item:: LoRA.cpp
            :name: loracpp

            .. code-block:: cpp

               #include "openvino/genai/text2image/pipeline.hpp"

               #include "imwrite.hpp"

               int32_t main(int32_t argc, char* argv[]) try {
                   OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]]");

                   const std::string models_path = argv[1], prompt = argv[2];
                   const std::string device = "CPU";  // GPU, NPU can be used as well

                   ov::genai::AdapterConfig adapter_config;
                   for(size_t i = 0; i < (argc - 3)/2; ++i) {
                       ov::genai::Adapter adapter(argv[3 + 2*i]);
                       float alpha = std::atof(argv[3 + 2*i + 1]);
                       adapter_config.add(adapter, alpha);
                   }

                   ov::genai::Text2ImagePipeline pipe(models_path, device, ov::genai::adapters(adapter_config));

                   std::cout << "Generating image with LoRA adapters applied, resulting image will be in lora.bmp\n";
                   ov::Tensor image = pipe.generate(prompt,
                       ov::genai::random_generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
                       ov::genai::width(512),
                       ov::genai::height(896),
                       ov::genai::num_inference_steps(20));
                   imwrite("lora.bmp", image, true);

                   std::cout << "Generating image without LoRA adapters applied, resulting image will be in baseline.bmp\n";
                   image = pipe.generate(prompt,
                       ov::genai::adapters(),
                       ov::genai::random_generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
                       ov::genai::width(512),
                       ov::genai::height(896),
                       ov::genai::num_inference_steps(20));
                   imwrite("baseline.bmp", image, true);

                   return EXIT_SUCCESS;
               } catch (const std::exception& error) {
                   try {
                       std::cerr << error.what() << '\n';
                   } catch (const std::ios_base::failure&) {}
                   return EXIT_FAILURE;
               } catch (...) {
                   try {
                       std::cerr << "Non-exception object thrown\n";
                   } catch (const std::ios_base::failure&) {}
                   return EXIT_FAILURE;
               }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/text2image/README.md>`__





Using GenAI in Speech Recognition
#################################


The application, shown in code samples below, performs inference on speech
recognition Whisper Models. The samples include the ``WhisperPipeline`` class
and use audio files in WAV format at a sampling rate of 16 kHz as input.

.. tab-set::

   .. tab-item:: Python
      :sync: cpp

      .. code-block:: python

         import openvino_genai
         import librosa


         def read_wav(filepath):
             raw_speech, samplerate = librosa.load(filepath, sr=16000)
             return raw_speech.tolist()


         def infer(model_dir: str, wav_file_path: str):
             raw_speech = read_wav(wav_file_path)
             pipe = openvino_genai.WhisperPipeline(model_dir)

             def streamer(word: str) -> bool:
                 print(word, end="")
                 return False

             result = pipe.generate(
                 raw_speech,
                 max_new_tokens=100,
                 language="<|en|>",
                 task="transcribe",
                 return_timestamps=True,
                 streamer=streamer,
             )

             print()
             for chunk in result.chunks:
                 print(f"timestamps: [{chunk.start_ts}, {chunk.end_ts}] text: {chunk.text}")


      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/whisper_speech_recognition/README.md>`__.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "audio_utils.hpp"
         #include "openvino/genai/whisper_pipeline.hpp"

         int main(int argc, char* argv[]) try {
             if (3 > argc) {
                 throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\"");
             }

             std::filesystem::path models_path = argv[1];
             std::string wav_file_path = argv[2];
             std::string device = "CPU"; // GPU can be used as well

             ov::genai::WhisperPipeline pipeline(models_path, device);

             ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

             ov::genai::WhisperGenerationConfig config(models_path / "generation_config.json");
             config.max_new_tokens = 100;
             config.language = "<|en|>";
             config.task = "transcribe";
             config.return_timestamps = true;

             auto streamer = [](std::string word) {
                 std::cout << word;
                 return false;
             };

             auto result = pipeline.generate(raw_speech, config, streamer);

             std::cout << "\n";

             for (auto& chunk : *result.chunks) {
                 std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
             }

         } catch (const std::exception& error) {
             try {
                 std::cerr << error.what() << '\n';
             } catch (const std::ios_base::failure&) {
             }
             return EXIT_FAILURE;
         } catch (...) {
             try {
                 std::cerr << "Non-exception object thrown\n";
             } catch (const std::ios_base::failure&) {
             }
             return EXIT_FAILURE;
         }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/whisper_speech_recognition/README.md>`__.


Using GenAI in Chat Scenario
############################

For chat scenarios where inputs and outputs represent a conversation, maintaining KVCache across inputs
may prove beneficial. The ``start_chat`` and ``finish_chat`` chat-specific methods are used to
mark a conversation session, as shown in the samples below:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai


         def streamer(subword):
             print(subword, end='', flush=True)
             return False


         def infer(model_dir: str):
             device = 'CPU'  # GPU can be used as well.
             pipe = openvino_genai.LLMPipeline(model_dir, device)

             config = openvino_genai.GenerationConfig()
             config.max_new_tokens = 100

             pipe.start_chat()
             while True:
                 try:
                     prompt = input('question:\n')
                 except EOFError:
                     break
                 pipe.generate(prompt, config, streamer)
                 print('\n----------')
             pipe.finish_chat()



      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/chat_sample/README.md>`__.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "openvino/genai/llm_pipeline.hpp"

         int main(int argc, char* argv[]) try {
             if (2 != argc) {
                 throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
             }
             std::string prompt;
             std::string models_path = argv[1];

             std::string device = "CPU";  // GPU, NPU can be used as well
             ov::genai::LLMPipeline pipe(models_path, device);

             ov::genai::GenerationConfig config;
             config.max_new_tokens = 100;
             std::function<bool(std::string)> streamer = [](std::string word) {
                 std::cout << word << std::flush;
                 return false;
             };

             pipe.start_chat();
             std::cout << "question:\n";
             while (std::getline(std::cin, prompt)) {
                 pipe.generate(prompt, config, streamer);
                 std::cout << "\n----------\n"
                     "question:\n";
             }
             pipe.finish_chat();
         } catch (const std::exception& error) {
             try {
                 std::cerr << error.what() << '\n';
             } catch (const std::ios_base::failure&) {}
             return EXIT_FAILURE;
         } catch (...) {
             try {
                 std::cerr << "Non-exception object thrown\n";
             } catch (const std::ios_base::failure&) {}
             return EXIT_FAILURE;
         }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/chat_sample/README.md>`__

Additional Resources
#####################

* :doc:`Install OpenVINO GenAI <../../../get-started/install-openvino/install-openvino-genai>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO GenAI Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
