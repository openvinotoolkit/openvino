module.exports = {
  rules: {
    'semi': ['error'],
    'no-var': ['error'],
    'max-len': ['error', { 'ignoreUrls': true }],
    'eol-last': ['error'],
    'indent': ['error', 2],
    'camelcase': ['error'],
    'semi-spacing': ['error'],
    'arrow-spacing': ['error'],
    'comma-spacing': ['error'],
    'no-multi-spaces': ['error'],
    'quotes': ['error', 'single'],
    'no-trailing-spaces': ['error'],
    'space-before-blocks': ['error'],
    'newline-before-return': ['error'],
    'comma-dangle': ['error', 'always-multiline'],
    'space-before-function-paren': ['error', {
      named: 'never',
      anonymous: 'never',
      asyncArrow: 'always'
    }],
    'key-spacing': ['error', { beforeColon: false }],
    'no-multiple-empty-lines': ['error', { max: 1, maxBOF: 0, maxEOF: 0 }],
    'keyword-spacing': ['error', { overrides: { catch: { after: false } } }],
    'prefer-destructuring': ["error", { "object": true, "array": false }],
    '@typescript-eslint/no-var-requires': 0,
  }
};
