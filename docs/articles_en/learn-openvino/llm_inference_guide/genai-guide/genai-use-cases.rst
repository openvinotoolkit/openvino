GenAI Use Cases
=====================

This article provides several use case scenarios for Generative AI model
inference. The applications presented in the code samples below
only require minimal configuration, like setting an inference device. Feel free
to explore and modify the source code as you need.


Using GenAI for Text-to-Image Generation
################################################

Examples below demonstrate inference on text-to-image models, like Stable Diffusion
1.5, 2.1, and LCM, with a text prompt as input. The :ref:`main.cpp <maincpp>`
sample shows basic usage of the ``Text2ImagePipeline`` pipeline.
:ref:`lora.cpp <loracpp>` shows how to apply LoRA adapters to the pipeline.


.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: main.cpp
            :name: maincpp

            .. code-block:: cpp

               int32_t main(int32_t argc, char* argv[]) try {
                   OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

                   const std::string models_path = argv[1], prompt = argv[2];
                   const std::string device = "CPU";  // GPU, NPU can be used as well.

                   ov::genai::Text2ImagePipeline pipe(models_path, device);
                   ov::Tensor image = pipe.generate(prompt,
                       ov::genai::width(512),
                       ov::genai::height(512),
                       ov::genai::num_inference_steps(20),
                       ov::genai::num_images_per_prompt(1));

                   // Saves images with a `num_images_per_prompt` name pattern.
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

               int32_t main(int32_t argc, char* argv[]) try {
                   OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]]");

                   const std::string models_path = argv[1], prompt = argv[2];
                   const std::string device = "CPU";  // GPU, NPU can be used as well.

                   ov::genai::AdapterConfig adapter_config;
                   // Applying Multiple LoRA adapters simultaneously is supported. Parse them all and the corresponding alphas from cmd parameters:
                   for(size_t i = 0; i < (argc - 3)/2; ++i) {
                       ov::genai::Adapter adapter(argv[3 + 2*i]);
                       float alpha = std::atof(argv[3 + 2*i + 1]);
                       adapter_config.add(adapter, alpha);
                   }

                   // LoRA adapters passed to the constructor will be activated by default in the next generation.
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
                       ov::genai::adapters(),  // Passing adapters as generation overrides set in the constructor; adapters() means no adapters.
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

         import argparse
         import openvino_genai
         import librosa


         def read_wav(filepath):
             raw_speech, samplerate = librosa.load(filepath, sr=16000)
             return raw_speech.tolist()


         def main():
             parser = argparse.ArgumentParser()
             parser.add_argument("model_dir")
             parser.add_argument("wav_file_path")
             args = parser.parse_args()

             raw_speech = read_wav(args.wav_file_path)

             pipe = openvino_genai.WhisperPipeline(args.model_dir)

             def streamer(word: str) -> bool:
                 print(word, end="")
                 return False

             result = pipe.generate(
                 raw_speech,
                 max_new_tokens=100,
                 # 'task' and 'language' parameters are supported for multilingual models only
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

         int main(int argc, char* argv[]) try {
             if (3 > argc) {
                 throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\"");
             }

             std::string model_path = argv[1];
             std::string wav_file_path = argv[2];

             ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

             ov::genai::WhisperPipeline pipeline{model_path};

             ov::genai::WhisperGenerationConfig config{model_path + "/generation_config.json"};
             config.max_new_tokens = 100;
             // 'task' and 'language' parameters are supported for multilingual models only
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

         import openvino_genai as ov_genai
         pipe = ov_genai.LLMPipeline(model_path)

         pipe.set_generation_config({'max_new_tokens': 100)

         pipe.start_chat()
         while True:
            print('question:')
            prompt = input()
            if prompt == 'Stop!':
               break
            print(pipe.generate(prompt))
         pipe.finish_chat()


      For more information, refer to the
      `Python sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/chat_sample/README.md>`__.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         int main(int argc, char* argv[]) {
            std::string prompt;

            std::string model_path = argv[1];
            ov::genai::LLMPipeline pipe(model_path, "CPU");

            ov::genai::GenerationConfig config = pipe.get_generation_config();
            config.max_new_tokens = 100;
            pipe.set_generation_config(config)

            pipe.start_chat();
            for (size_t i = 0; i < questions.size(); i++) {
               std::cout << "question:\n";
               std::getline(std::cin, prompt);

               std::cout << pipe.generate(prompt) << std::endl;
            }
            pipe.finish_chat();
         }


      For more information, refer to the
      `C++ sample <https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/chat_sample/README.md>`__

Additional Resources
#####################

* :doc:`Install OpenVINO GenAI <../../../get-started/install-openvino/install-openvino-genai>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO GenAI Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
