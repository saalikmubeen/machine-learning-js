require("@tensorflow/tfjs-node");
const LinearRegression = require("./linear-regression");
const loadCSV = require("../load-csv");

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
