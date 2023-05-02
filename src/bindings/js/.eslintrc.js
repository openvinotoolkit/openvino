module.exports = {
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended'],
  ignorePatterns: ['**/*.js', 'node_modules/', 'types/', 'dist/', 'bin/'],
  root: true,
  rules: {
    'semi': ['error'],
    'max-len': ['error'],
    'eol-last': ['error'],
    'indent': ['error', 2],
    'camelcase': ['error'],
    'comma-dangle': ['error', 'always-multiline'],
    'no-multiple-empty-lines': ['error', { max: 1, maxBOF: 0, maxEOF: 0 }],
  }
};
