const fs = require('fs');

exports.cleanWord = cleanWord;
exports.encodeByVoc = encodeByVoc;
exports.textToTokens = textToTokens;
exports.splitToWords = splitToWords;
exports.loadVocabFile = loadVocabFile;

// load vocabulary file for encoding
function loadVocabFile(vocabFileName) {
  const vocab = {};
  const lines = fs.readFileSync(vocabFileName, 'utf-8').split('\n');

  lines.forEach((line, index) => {
    const token = line.trim();

    vocab[token] = index;
  });

  return vocab;
}

// remove mark and control chars
function cleanWord(w) {
  let wo = ''; // accumulator for output word

  for (const c of w.normalize('NFD')) {
    const charCode = c.charCodeAt(0);
    // remove mark nonspacing code and controls
    if (charCode < 32 || charCode == 127) continue;

    wo += c;
  }

  return wo;
}

// split word by vocab items and get tok codes
// iteratively return codes
function encodeByVoc(w, vocab) {
  w = cleanWord(w);
  const res = [];
  const wordIndexes = splitToWords(w);

  for (let el of wordIndexes) {
    const [s0, e0] = el;

    let s = s0;
    let e = e0;
    const tokens = [];

    while (e > s) {
      const subword = s == s0 ? w.slice(s, e) : '##' + w.slice(s, e);

      if (vocab[subword]) {
        tokens.push(vocab[subword]);
        s = e;
        e = e0;
      }
      else {
        e -= 1;
      }
    }

    if (s < e0) tokens.push(vocab['[UNK]']);

    res.push(...tokens);
  }

  return res;
}

// split big text into words by spaces
// return start and end indexes of words
function splitToWords(text) {
  let start;
  let prevIsSep = true; // mark initial prev as space to start word from 0 char
  const result = [];

  for (let i = 0; i < text.length + 1; i++) {
    const c = text[i] || ' ';
    const isPunc = /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/.test(c);
    const curIsSep = c.trim() === '' || isPunc;

    if (prevIsSep !== curIsSep)
      if (prevIsSep)
        start = i;
      else {
        result.push([start, i]);
        prevIsSep = curIsSep;
      }

    if (isPunc) result.push([i, i + 1]);

    prevIsSep = curIsSep;
  }

  return result;
}

// get big text and return list of token id and start-end positions
// for each id in original texts
function textToTokens(text, vocab) {
  const tokensId = [];
  const tokensSe = [];
  const wordIndices = splitToWords(text);

  for (const [start, end] of wordIndices) {
    const word = text.slice(start, end);
    const encodedTokens = encodeByVoc(word, vocab);

    for (const token of encodedTokens) {
      tokensId.push(token);
      tokensSe.push([start, end]);
    }
  }

  return [tokensId, tokensSe];
}
