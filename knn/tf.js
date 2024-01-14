require("@tensorflow/tfjs-node"); // to run the calculations the CPU
const tf = require("@tensorflow/tfjs");

// 1 Dimensional Tensor:
let t_0 = tf.tensor([1, 2, 3]);

// 2 Dimensional Tensor:
let t_2 = tf.tensor([
	[1, 2],
	[3, 4],
]);

// 3 Dimensional Tensor:
let t_3 = tf.tensor([
	[
		[1, 2],
		[3, 4],
	],
	[
		[5, 6],
		[7, 8],
	],
]);

// An easy way to determine the number of dimensions of a tensor is to count the number of opening square brackets
// before you see a number.

// ** Shape of a Tensor:
// How many elements there are in each dimension of a tensor.
// Imagine calling ".length" once on each dimension of a tensor from outside to inside.
// [2, 4, 1] -> lenghth of arrays from outside to inside
// e.g

let t = tf.tensor([1, 2, 3, 4]); // shape is [4]

t = tf.tensor([
	[1, 2, 3],
	[3, 4, 5],
]); // shape is [2, 3]  (2 rows, 3 columns) | [#rows, #cols]

t = tf.tensor([
	[
		[1, 2, 4, 5],
		[3, 4, 5, 6],
	],
]); // shape is [1, 2, 4] (1 row, 2 columns, 4 depth) | [#rows, #cols, #depth]

console.log(t.shape);

// ** Element wise operations:
// When we do elementwise operation, we look at identical indicies or positions in each tensor and perform the operation
// on those two individual elements and then place the result in the same index in new the output tensor.
// To do elementwise operations, the shapes of the tensors must be identical. If they are not, we get an error.

// Addition:
let t1 = tf.tensor([1, 2, 3]); // shape = [3]
let t2 = tf.tensor([4, 5, 6]); // shape = [3]

t1.add(t2).print(); // [5, 7, 9]

// If shapes of two
t1 = tf.tensor([
	[1, 2],
	[3, 4],
]); // shape = [2, 2]

t2 = tf.tensor([
	[5, 6, 3],
	[7, 8, 3],
]); // shape = [2, 3]

// t1.add(t2).print(); // Message: Incompatible shapes: [2,2] vs. [2,3]

// ** Broadcasting:
// Shapes to match for elementwise operations is not a hard and fast rule. There are corner cases where we can do
// element wise operations on tensors even if their shapes are not identical. This is called broadcasting.
// The idea is to automatically expand the smaller tensor to match the shape of the larger tensor so that
// element-wise operations can be performed without explicitly duplicating the smaller tensor.
// Broadcasting is a set of rules that TensorFlow uses to determine how it will do elementwise operations on tensors
// with different shapes. The rules are as follows:

// 1. Dimensions must be compatible:
//    Broadcasting is only possible when the dimensions of the arrays involved in the operation are compatible.
//    Dimensions are compatible when they are equal or one of them is 1 while going from right to left in the
//    shape arrays.

// 2. Dimensions are padded with ones on the left:
//   If the shapes of the input arrays differ, the shape of the tensor with fewer dimensions is padded with
//   ones on its leading (left) side..

// 3. Arrays are broadcasted along the missing dimensions:
//    The size-1 dimensions in the smaller array are broadcasted along the corresponding dimensions in the larger array.
//    In other words, the smaller array is repeated along the last dimension to match the shape of the larger array.
//    Or
//    If the shape of the two tensors does not match in any dimension, the tensor with shape equal to 1 in that
//    dimension is stretched to match the other shape.

// 4. If two dimensions are different and not equal to 1, an error occurs:
//    If two dimensions are incompatible (neither is 1), and they are not equal, the operation is not valid,
//    and an error is raised.

// Consider the following example:
// Define a matrix (2x3)
const matrix = tf.tensor2d([
	[1, 2, 3],
	[4, 5, 6],
]);

// Define a vector (3 elements)
const vector = tf.tensor1d([10, 20, 30]);

// Perform broadcasting to add the vector to each row of the matrix
const result = matrix.add(vector);

// Print the result
result.print();
// [
// 	[11, 22, 33],
// 	[14, 25, 36],
// ];

// In this example:

// The matrix has a shape of (2, 3) (2 rows, 3 columns).
// The vector has a shape of (3,) (1 dimension with 3 elements).
// The key point here is that the smaller tensor (vector) is padded with ones on its left side to match
// the number of dimensions of the larger tensor (matrix). The resulting broadcasted shape of the vector becomes (1, 3).
// The operation matrix.add(vector) then adds the broadcasted vector to each row of the matrix.

// ** Tensor Accessors:

t = tf.tensor([
	[5, 6, 3],
	[7, 8, 3],
]); // shape = [2, 3

console.log(t.arraySync()[0][1]); // t.arraySync()[rowIdx][colIdx]

// Slicing a tensor:

t = tf.tensor([
	[10, 20, 30, 40],
	[50, 60, 70, 80],
	[90, 100, 110, 120],
	[130, 140, 150, 160],
	[170, 180, 190, 200],
]);

// t.slice([rowIdx, colIdx], [numOfRows, numOfCols])
// [rowIdx, colIdx] -> starting point, [numOfRows, numOfCols] -> how many rows and cols to slice (size)
t.slice([0, 1], [1, 2]).print(); // [[20, 30],]
t.slice([1, 0], [3, 2]).print(); // [[50, 60], [90, 100], [130, 140]]
t.slice([1, 1], [-1, 1]).print(); // [[60], [100], [140], [180]] | -1 means to the end

// ** Concatenating Tensors:

// Concatenate along the rows (axis=0) (axis 0 can be thought of as the vertical axis or Y axis)
// Along the rows means that we are adding more rows to the tensor, adding rows while walking along the
// vertical axis.
t1 = tf.tensor([
	[1, 2, 3],
	[4, 5, 6],
]);

t2 = tf.tensor([
	[7, 8, 9],
	[10, 11, 12],
]);

t1.concat(t2, (axis = 0)).print();
//  [
// 		[1, 2, 3],
// 		[4, 5, 6],
// 		[7, 8, 9],
// 		[10, 11, 12],
//  ];

// Concatenate along the columns (axis=1) (axis 1 can be thought of as the horizontal axis or X axis)
// Along the columns means that we are adding more columns to the tensor, adding columns while walking along the
// horizontal axis.
t1.concat(t2, (axis = 1)).print();
// [
// 	[1, 2, 3, 7, 8, 9],
// 	[4, 5, 6, 10, 11, 12],
// ];

// While concating tensors, the shapes of the tensors must be identical except for the dimension along which
// they are being concatenated. For example, if we are concatenating along the rows, the number of columns
// in both tensors must be the same. If we are concatenating along the columns, the number of rows in both
// tensors must be the same.

// ** Summing along an axis:

t = tf.tensor([
	[1, 2, 3],
	[4, 5, 6],
]);

t.sum().print(); // 21 | sum of all elements in the tensor

t.sum((axis = 0)).print(); // [5, 7, 9] | sum along the rows (axis=0) | sum of each column

t.sum((axis = 1)).print(); // [6, 15] | sum along the columns (axis=1) | sum of each row

// ** Expanding Dimensions:
