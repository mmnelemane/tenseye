features and model put together will give us a label
features == input
label == output

training data with all features and all labels for these features

test data will have features

Tensors

- a vector generalized to a higher dimension

- a vector is a datapoint

Each tensor has a datatype and shape
- shape - representation based on dimension

string = tf.variable("this is a string", tf.string)
number = tf.variable(124, tf.int16)


Rank/Degree of tensors
-  number of dimensions involved in a tensor


Types of tensors
- Variable - mutable type
- Constant
- PlaceHolder
- SparseTensor


Core Learning Algorithms with Tensorflow

Categories : Linear Regression, Classification, Clustering, Hidden Markov Models

1. Linear Regression
- data that correlates linearly
  y = mx + b
  m -> slope
  b -> y-intercept

  As close to data points as possible (same no of datapoints on both sides of the line for a best fit)
  
  categorical data - non-numeric data


Training Process:
- feeding data in batches (e.g: 32 entries at a time)

epochs - no of times the model will see the same data during training
       - each time in a different order

overfitting - feeding data too many times leading to memorizing the data

input function - breaking data into batches and epochs (encoding into tf.data.Dataset object)






