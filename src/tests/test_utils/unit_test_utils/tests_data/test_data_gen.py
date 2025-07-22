import tensorflow as tf
import numpy as np

def format_array(arr):
    """Format numpy array as comma-separated values for C++ macro."""
    if isinstance(arr, (list, np.ndarray)):
        # Special handling for 2D indices
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            # Flatten 2D list into 1D
            flattened = [val for sublist in arr for val in sublist]
            return ', '.join(str(x) for x in flattened)
        
        # Flatten multi-dimensional arrays
        if hasattr(arr, 'flatten'):
            arr = arr.flatten()
        return ', '.join(str(x) for x in arr)
    else:
        return str(arr)

def generate_test_case(indices, values, dense_shape, default_value, name):
    """Generate a test case using TensorFlow's SparseFillEmptyRows."""
    # Make sure eager execution is enabled (it's default in TF 2.x)
    tf.config.run_functions_eagerly(True)
    
    # Convert inputs to appropriate TensorFlow tensors
    # Handle empty indices correctly by ensuring it's a 2D tensor with shape [0, 2]
    if isinstance(indices, list) and len(indices) == 0:
        tf_indices = tf.zeros((0, 2), dtype=tf.int64)
        tf_values = tf.zeros((0,), dtype=tf.float32)
    else:
        tf_indices = tf.constant(indices, dtype=tf.int64)
        tf_values = tf.constant(values, dtype=tf.float32)
    
    tf_dense_shape = tf.constant(dense_shape, dtype=tf.int64)
    tf_default_value = tf.constant(42, dtype=tf.float32)
    
    # Create a SparseTensor
    sparse_tensor = tf.sparse.SparseTensor(
        indices=tf_indices,
        values=tf_values,
        dense_shape=tf_dense_shape
    )
    
    # Run SparseFillEmptyRows using the high-level API
    output_st, empty_row_indicator = tf.sparse.fill_empty_rows(
        sparse_tensor, tf_default_value
    )
    
    # Extract the outputs
    out_indices = output_st.indices.numpy()
    out_values = output_st.values.numpy()
    out_empty_row_indicator = empty_row_indicator.numpy().astype(int)  # Convert boolean to int
    
    # Format the test case for C++
    test_case = f"""TEST_DATA(LIST({format_array(indices)}),
          LIST({format_array(values)}),
          LIST({format_array(dense_shape)}),
          LIST({format_array(out_indices)}),
          LIST({format_array(out_values)}),
          LIST({format_array(out_empty_row_indicator)}),
          "{name}")"""
    
    return test_case

def main():
    # Test case 1: Basic case with empty rows
    print(generate_test_case(
        indices=[[0, 1], [0, 3], [2, 2]], 
        values=[1, 2, 3], 
        dense_shape=[3, 5], 
        default_value=42, 
        name="BasicCase"
    ))
    
    # Test case 2: No empty rows
    print(generate_test_case(
        indices=[[0, 0], [1, 1], [2, 2]], 
        values=[10, 20, 30], 
        dense_shape=[3, 3], 
        default_value=42, 
        name="NoEmptyRows"
    ))
    
    # Test case 3: All empty rows
    print(generate_test_case(
        indices=[], 
        values=[], 
        dense_shape=[3, 2], 
        default_value=42, 
        name="AllEmptyRows"
    ))
    
    # Test case 4: Empty rows at beginning and end
    print(generate_test_case(
        indices=[[1, 0], [1, 1]], 
        values=[5, 6], 
        dense_shape=[3, 3], 
        default_value=42, 
        name="EmptyRowsAtBothEnds"
    ))
    
    # Test case 5: Larger dimensions
    print(generate_test_case(
        indices=[[0, 2], [2, 0], [4, 1], [6, 3]], 
        values=[1.5, 2.5, 3.5, 4.5], 
        dense_shape=[8, 4], 
        default_value=42, 
        name="LargerDimensions"
    ))
    
    # Test case 6: Huge number of rows (sparse)
    row_count = 10000
    print(generate_test_case(
        indices=[[100, 0], [1000, 1], [5000, 2], [9000, 3]], 
        values=[1.1, 2.2, 3.3, 4.4], 
        dense_shape=[row_count, 5], 
        default_value=42, 
        name="HugeRowCount"
    ))
    
    # Test case 7: Huge number of columns (sparse)
    col_count = 10000
    print(generate_test_case(
        indices=[[0, 100], [1, 1000], [2, 5000], [3, 9000]], 
        values=[5.5, 6.6, 7.7, 8.8], 
        dense_shape=[5, col_count], 
        default_value=42, 
        name="HugeColumnCount"
    ))
    
    # Test case 8: Large number of non-zero entries (still sparse)
    entries = 1000
    indices = [[i % 100, i // 100] for i in range(entries)]
    values = [float(i) for i in range(entries)]
    print(generate_test_case(
        indices=indices, 
        values=values, 
        dense_shape=[100, 100], 
        default_value=42, 
        name="ManyEntries"
    ))
    
    # Test case 9: Huge output - mostly empty tensor with scattered values
    print(generate_test_case(
        indices=[[i*50, i] for i in range(10)],  # Values at rows 0, 500, 1000, etc.
        values=[i*10.0 for i in range(10)],
        dense_shape=[500, 20],
        default_value=42,
        name="HugeOutput"
    ))

if __name__ == "__main__":
    main()
