.. {#openvino_docs_ops_internal_NGramHashFunction}

NGramHashFunction
=================


.. meta::
  :description: Learn about NGramHashFunction, which hashes prepared n-gram windows into embedding table row indices.

**Versioned name**: *NGramHashFunction*

**Category**: *Sequence processing*

**Short description**: Computes embedding table row indices for each n-gram window and hash head.

**Detailed description**

The last input dimension contains ``ngram_size`` token IDs, oldest first. For each window, let
``x[k] = ngrams[..., ngram_size - 1 - k]``, so ``x[0]`` is the current token.

Orders ``2 .. ngram_size`` are hashed independently for each window. Output heads are grouped by
ascending order, with ``heads_per_order[o - 2]`` heads for order ``o``:

.. code-block:: py

    mixed = x[0] * multipliers[0]
    h = 0
    for k in range(1, ngram_size):
        mixed ^= x[k] * multipliers[k]
        for _ in range(heads_per_order[k - 1]):
            row_ids[..., h] = floor_mod(mixed, head_vocab_sizes[h]) + head_offsets[h]
            h += 1

Arithmetic uses signed 64-bit two's complement values, with multiplication wrapping modulo ``2^64``.
An int32 input is sign-extended before hashing. ``floor_mod`` returns a non-negative remainder for the
positive divisor, including when ``mixed`` is negative.

The operation is stateless and applies independently over all leading dimensions. Window preparation,
padding and sequence boundaries are handled outside the operation.


**Attributes**

* *ngram_size*

  * **Description**: the largest n-gram order and the size of the last dimension of ``ngrams``.
    Orders ``2 .. ngram_size`` are hashed.
  * **Range of values**: an integer ``>= 2``
  * **Type**: ``int``
  * **Required**: *yes*

* *heads_per_order*

  * **Description**: the number of hashed heads of every n-gram order ``2 .. ngram_size``,
    ``ngram_size - 1`` values, where element ``heads_per_order[o - 2]`` is the head count for order ``o``.
    A value of ``0`` skips that order. ``H = sum(heads_per_order)`` must be positive.
  * **Range of values**: non-negative integers
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *multipliers*

  * **Description**: the ``ngram_size`` hash multipliers; ``multipliers[k]`` applies to the token ``k``
    steps back (``multipliers[0]`` to the current token).
  * **Range of values**: any ``int64``
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *head_vocab_sizes*

  * **Description**: the vocabulary size (modulus) of every hashed head, ``H`` values, grouped by order.
  * **Range of values**: integers in ``[1, 2^31 - 1]``, with
    ``head_offsets[h] + head_vocab_sizes[h] - 1 <= 2^31 - 1``
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *head_offsets*

  * **Description**: the first row of every hashed head in the packed table, ``H`` values.
  * **Range of values**: non-negative integers, ``head_offsets[h] + head_vocab_sizes[h] - 1 <= 2^31 - 1``
  * **Type**: ``int64[]``
  * **Required**: *yes*


**Inputs**

* **1**: ``ngrams`` - tensor of type *T_IND* and shape ``[..., ngram_size]``: the n-gram window of
  every token, oldest first, the current token last. The rank is at least 1. **Required.**


**Outputs**

* **1**: ``row_ids`` - tensor of type ``int32`` and shape ``[..., H]``: the row of the packed n-gram
  table for every token and hashed head.


**Types**

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``...`` - arbitrary leading dimensions, preserved in the output.

* ``H`` - number of hashed heads, ``sum(heads_per_order)``.


**Example**

For ``ngrams = [[1, 2, 3], [4, 5, 6]]``, the following configuration produces
``row_ids = [[3, 15], [0, 21]]``.

.. code-block:: xml
   :force:

    <layer id="1" name="ngram_hash" type="NGramHashFunction" version="extension">
      <data ngram_size="3" heads_per_order="1,1" multipliers="3,5,7"
          head_vocab_sizes="11,13" head_offsets="0,11"/>
        <input>
        <port id="0" precision="I64">
          <dim>2</dim>
          <dim>3</dim>
            </port>
        </input>
        <output>
        <port id="1" precision="I32">
          <dim>2</dim>
          <dim>2</dim>
            </port>
        </output>
    </layer>
