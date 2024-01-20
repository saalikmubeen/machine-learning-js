require("@tensorflow/tfjs-node");
const LinearRegression = require("./linear-regression");
const loadCSV = require("../load-csv");
// const plot = require("node-remote-plot");

// Miles Per Gallon (mpg) = m * (Horsepower) + b

let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
	shuffle: true,
	splitTest: 50, // 50 rows for testing
	dataColumns: ["horsepower", "weight", "displacement"], // "weight", "displacement"
	labelColumns: ["mpg"],
});

const linearRegression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iterations: 5,
	batchSize: 10,
});

linearRegression.train();
// linearRegression.test(testFeatures, testLabels)

linearRegression
	.predict([
		[120, 2, 380], // [horsepower, weight, displacement]
		[135, 2.1, 420],
	])
	.print();

// Learning Curve:
// The learning curve is a plot of the cost function's value (J) (y axis) and the number of iterations (x axis)
// of the training algorithm. The learning curve is useful for diagnosing bias/variance problems.
// Looking at this graph helps you to see how your cost J changes after each iteration of gradient descent.
// If gradient descent is working properly, then the cost J should decrease after every single iteration.
// if you plot the cost for a number of iterations and notice that the costs sometimes goes up and sometimes
// goes down, you should take that as a clear sign that gradient descent is not working properly.
// This could mean that there's a bug in the code. Or sometimes it could mean that your learning rate is too large.
// Looking at this learning curve, you can try to spot whether or not gradient descent is converging.
// By the way, the number of iterations that gradient descent takes a conversion can vary a lot between
// different applications. In one application, it may converge after just 30 iterations.
// For a different application, it could take 1,000 or 100,000 iterations

// Automatic Convergence Test:
// The automatic convergence test is a way to automatically stop the gradient descent algorithm when
// it's no longer making rapid progress. This is a useful way to speed up the training process.
// Here is the Greek alphabet epsilon (ε). Let's let epsilon be a variable representing a small number, such as
// 0.001 or 10^-3. If the cost J decreases by less than this number epsilon on one iteration, then you're likely
// on this flattened part of the curve that you see on the left and you can declare convergence.
// Remember, convergence, hopefully in the case that you found parameters w and b that are close to the
// minimum possible value of J.

// plot({
// 	x: logisticRegression.costHistory.reverse(),
// });
