# Code Style Guide

This document outlines the coding standards and guidelines of javascript and typescript part of **openvino-node** package. These rules help maintain code quality and consistency throughout the codebase.

Make sure your IDE has ESLint plugin installed. Its rules are specified in the [.eslint-global.js file](../.eslintrc-global.js). Keep in mind that your PR will not be approved if it does not meet the following requirements.

## General Rules

### 1. Semicolons
- **Rule**: Always use semicolons at the end of statements.
- **Enforced By**: `semi: ['error']`

### 2. Variable Declarations
- **Rule**: Use `let` or `const` instead of `var`.
- **Enforced By**: `no-var: ['error']`

### 3. Line Length
- **Rule**: Lines should not exceed the maximum length.
- **Enforced By**: `max-len: ['error']`

### 4. End of Line
- **Rule**: Ensure that files end with a newline.
- **Enforced By**: `eol-last: ['error']`

### 5. Indentation
- **Rule**: Use 2 spaces for indentation.
- **Enforced By**: `indent: ['error', 2]`

### 6. Naming Conventions
- **Rule**: Use camelCase for variable and function names.
- **Enforced By**: `camelcase: ['error']`

### 7. Semicolon Spacing
- **Rule**: Enforce spacing around semicolons.
- **Enforced By**: `semi-spacing: ['error']`

### 8. Arrow Function Spacing
- **Rule**: Enforce spacing around arrow functions.
- **Enforced By**: `arrow-spacing: ['error']`

### 9. Comma Spacing
- **Rule**: Enforce spacing around commas.
- **Enforced By**: `comma-spacing: ['error']`

### 10. Multiple Spaces
- **Rule**: Disallow multiple spaces except for indentation.
- **Enforced By**: `no-multi-spaces: ['error']`

### 11. Quotes
- **Rule**: Use single quotes for strings.
- **Enforced By**: `quotes: ['error', 'single']`

### 12. Trailing Spaces
- **Rule**: Disallow trailing spaces at the end of lines.
- **Enforced By**: `no-trailing-spaces: ['error']`

### 13. Space Before Blocks
- **Rule**: Require space before blocks.
- **Enforced By**: `space-before-blocks: ['error']`

### 14. Newline Before Return
- **Rule**: Enforce newline before return statements.
- **Enforced By**: `newline-before-return: ['error']`

### 15. Comma Dangle
- **Rule**: Require trailing commas in multiline object literals.
- **Enforced By**: `comma-dangle: ['error', 'always-multiline']`

### 16. Space Before Function Parentheses
- **Rule**: Enforce consistent spacing before function parentheses.
  - Named functions: No space
  - Anonymous functions: No space
  - Async arrow functions: Space
- **Enforced By**: `space-before-function-paren: ['error', { named: 'never', anonymous: 'never', asyncArrow: 'always' }]`

### 17. Key Spacing in Object Literals
- **Rule**: Enforce consistent spacing between keys and values in object literals.
- **Enforced By**: `key-spacing: ['error', { beforeColon: false }]`

### 18. Multiple Empty Lines
- **Rule**: Disallow multiple empty lines.
  - Maximum empty lines: 1
  - Maximum at the beginning of file: 0
  - Maximum at the end of file: 0
- **Enforced By**: `no-multiple-empty-lines: ['error', { max: 1, maxBOF: 0, maxEOF: 0 }]`

### 19. Keyword Spacing
- **Rule**: Enforce consistent spacing around keywords.
  - Special case for `catch` keyword: No space after `catch`
- **Enforced By**: `keyword-spacing: ['error', { overrides: { catch: { after: false } } }]`

## Additional Resources

For further details on each rule, please refer to the [ESLint documentation](https://eslint.org/docs/rules/).
