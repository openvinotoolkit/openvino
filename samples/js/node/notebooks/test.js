const {
  reshape,
  getShape,
  extractValues,
  matrixMultiplication,
  triu,
  tril,
  argMax,
  arange,
  downloadFile,
} = require('./helpers.js');
const tokens = require('./tokens_bert.js');
const ov = require('../node_modules/openvinojs-node/build/Release/ov_node_addon.node');

main();

async function main() {
  const baseArtifactsDir = '../../assets/models';

  const modelName = 'bert-small-uncased-whole-word-masking-squad-int8-0002';
  const modelXMLName = `${modelName}.xml`;
  const modelBINName = `${modelName}.bin`;

  const modelXMLPath = baseArtifactsDir + '/' + modelXMLName;

  const baseURL = 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/bert-small-uncased-whole-word-masking-squad-int8-0002/FP16-INT8/';

  // await downloadFile(baseURL + modelXMLName, modelXMLName, baseArtifactsDir);
  // await downloadFile(baseURL + modelBINName, modelBINName, baseArtifactsDir);

  const core = new ov.Core();
  const model = core.read_model(modelXMLPath);
  const compiledModel = core.compile_model(model, 'CPU');

  const inputs = compiledModel.inputs;
  const outputs = compiledModel.outputs;

  const inputSize = compiledModel.input(0).shape[1];

  inputs.forEach(i => console.log(`${i}`));
  console.log('===')
  outputs.forEach(o => console.log(`${o}`));

  // The path to the vocabulary file.
  const vocabFilePath = "../../assets/text/vocab.txt";

  // Create a dictionary with words and their indices.
  const vocab = tokens.loadVocabFile(vocabFilePath);

  // Define special tokens.
  const clsToken = vocab["[CLS]"];
  const padToken = vocab["[PAD]"];
  const sepToken = vocab["[SEP]"];

  // A function to load text from given urls.
  function loadContext(sources) {
    // TODO: Add opportunity to parse content by URL
    const input_urls = [];
    const paragraphs = [];

    for (source of sources) {
      paragraphs.push(source);

      // Produce one big context string.
      return paragraphs.join('\n');
    }
  }

  // Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L188
  function findBestAnswerWindow(startScore, endScore, contextStartIdx, contextEndIdx) {
    const contextLen = contextEndIdx - contextStartIdx;

    const mat1 = reshape(startScore.slice(contextStartIdx, contextEndIdx), [contextLen, 1]);
    const mat2 = reshape(endScore.slice(contextStartIdx, contextEndIdx), [1, contextLen]);

    let scoreMat = matrixMultiplication(mat1, mat2);

    // Reset candidates with end before start.
    scoreMat = triu(scoreMat);
    // Reset long candidates (>16 words).
    scoreMat = tril(scoreMat, 16);

    // Find the best start-end pair.
    const coef = argMax(extractValues(scoreMat));
    const secondShapeDim = getShape(scoreMat)[1];

    const maxS = parseInt(coef/secondShapeDim);
    const maxE = coef%secondShapeDim;

    const maxScore = scoreMat[maxS][maxE];

    return [maxScore, maxS, maxE];
  }


  function getScore(logits) {
    const out = exp(logits);
    const summedRows = sumRows(out);

    return out.map(row => row.map((colEl, i) => colEl/summedRows[i]));
  }

  // Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L163
  function postprocess(outputStart, outputEnd, questionTokens, contextTokensStartEnd, padding, startIdx) {
    console.log('== postprocess');
    // Get start-end scores for the context.
    const scoreStart = getScore(outputStart);
    const scoreEnd = getScore(outputEnd);

    // An index of the first context token in a tensor.
    const contextStartIdx = questionTokens.length;
    // An index of the last+1 context token in a tensor.
    const contextEndIdx = inputSize - padding - 1;

    // Find product of all start-end combinations to find the best one.
    let [maxScore, maxStart, maxEnd] = findBestAnswerWindow(scoreStart,
                                                            scoreEnd,
                                                            contextStartIdx,
                                                            contextEndIdx);

    // Convert to context text start-end index.
    maxStart = contextTokensStartEnd[maxStart + startIdx][0];
    maxEnd = contextTokensStartEnd[maxEnd + startIdx][1];

    return maxScore, maxStart, maxEnd;
  }

  // A function to add padding.
  function pad({ inputIds, attentionMask, tokenTypeIds }) {
    // How many padding tokens.
    const diffInputSize = inputSize - inputIds.length;

    if (diffInputSize > 0) {
      // Add padding to all the inputs.
      inputIds = inputIds.concat(Array(diffInputSize).fill(padToken));
      attentionMask = attentionMask.concat(Array(diffInputSize).fill(0));
      tokenTypeIds = tokenTypeIds.concat(Array(diffInputSize).fill(0));
    }

    return [inputIds, attentionMask, tokenTypeIds, diffInputSize];
  }

  // A generator of a sequence of inputs.
  function* prepareInput(questionTokens, contextTokens) {
    console.log('== prepareInput');

    // A length of question in tokens.
    const questionLen = questionTokens.length;
    // The context part size.
    const contextLen = inputSize - questionLen - 3;

    if (contextLen < 16)
        throw new Error('Question is too long in comparison to input size. No space for context');

    const inputLayerNames = inputs.map(i => i.toString());

    // Take parts of the context with overlapping by 0.5.
    const max = Math.max(1, contextTokens.length - contextLen);

    console.log('== here');

    for (let start = 0; start < max; start += parseInt(contextLen / 2)) {
      console.log('== here');

      // A part of the context.
      const partContextTokens = contextTokens.slice(start, start + contextLen);
      // The input: a question and the context separated by special tokens.
      let inputIds = [clsToken, ...questionTokens, sepToken, ...partContextTokens, sepToken];
      // 1 for any index if there is no padding token, 0 otherwise.
      let attentionMask = Array(inputIds.length).fill(1);
      // 0 for question tokens, 1 for context part.
      let tokenTypeIds = [...Array(questionLen + 2).fill(0), ...Array(partContextTokens.length + 1).fill(1)];

      let padNumber = 0;

      // Add padding at the end.
      [inputIds, attentionMask, tokenTypeIds, padNumber] = pad({ inputIds, attentionMask, tokenTypeIds });

      // Create an input to feed the model.
      const inputDict = {
        'input_ids': new Int32Array(inputIds),
        'attention_mask': new Int32Array(attentionMask),
        'token_type_ids': new Int32Array(tokenTypeIds),
      };

      // Some models require additional position_ids.
      if (inputLayerNames.includes('position_ids')) {
        positionIds = inputIds.map((_, index) => index);
        inputDict['position_ids'] = new Int32Array(positionIds);
      }

      yield [inputDict, padNumber, start];
    }
  }

  function getBestAnswer(question, context) {
    // Convert the context string to tokens.
    const [contextTokens, contextTokensStartEnd] = tokens.textToTokens(context.toLowerCase(), vocab);
    // Convert the question string to tokens.
    const [questionTokens] = tokens.textToTokens(question.toLowerCase(), vocab);

    const results = [];
    // Iterate through different parts of the context.
    for ([networkInput, padding, startIdx] of prepareInput(questionTokens, contextTokens)) {
      // console.log({ networkInput, padding, startIdx });

      // Get output layers.
      const outputStartKey = compiledModel.output('output_s');
      const outputEndKey = compiledModel.output('output_e');

      // OpenVINO inference.
      const inferRequest = compiledModel.create_infer_request();
      // FIXME: implement supporting of multiinput model

      inferRequest.set_input_tensor(networkInput);
      inferRequest.infer();

      const resultStart = inferRequest.getTensor(outputStartKey);
      const resultEnd = inferRequest.getTensor(outputEndKey);

      // Postprocess the result, getting the score and context range for the answer.
      const scoreStartEnd = postprocess(resultStart[0], // ???
                                    resultEnd[0],         // ???
                                    questionTokens,
                                    contextTokensStartEnd,
                                    padding,
                                    startIdx);
      results.append(scoreStartEnd);
    }

    console.log({ results });

    // Find the highest score.
    const answer = Math.max(results);
    // Return the part of the context, which is already an answer.
    return context.slice(answer[1], answer[2]), answer[0];
  }

  function runQuestionAnswering(sources, exampleQuestion) {
    console.log(`Context: ${sources}`);
    const context = loadContext(sources);

    if (!context.length)
        return console.log('Error: Empty context or outside paragraphs');

    if (exampleQuestion) {
        const startTime = process.hrtime();
        const [answer, score] = getBestAnswer(exampleQuestion, context);
        const execTime = process.hrtime(startTime);

        console.log(`Question: ${exampleQuestion}`);
        console.log(`Answer: ${answer}`);
        console.log(`Score: ${score}`);
        console.log(`Time: ${execTime}s`);
    }
  }

  const textSources = ["Bert is a yellow Muppet character on the long running PBS and HBO children's television show Sesame Street. Bert was originally performed by Frank Oz."];

  try {
    const result = runQuestionAnswering(textSources, 'Who is Bert?');

    console.log(result);
  } catch(e) {
    console.log(e);
  }
}
