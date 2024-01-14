require("@tensorflow/tfjs-node");
const _ = require("lodash");
const loadCSV = require("../load-csv-2");
const LogisticRegression = require("./multinomail-logistic-regression");
// const plot = require("node-remote-plot");

// Given the horsepower, weight and displacement of the car, predict whether it will have a "low",
// "high" or "medium" fuel efficiency.

// "Fuel Efficiency" is not a feature or metric in our dataset. We will derive it ourselves
// from the mpg. If mile per gallon (mpg) is 0 - 15, fuel efficiency is "low", if mpg is 15-30, fuel efficiency
// is "medium" and if mpg is > 30, fuel efficiency is "high".

const { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
	dataColumns: ["horsepower", "displacement", "weight"],
	labelColumns: ["mpg"],
	shuffle: true,
	splitTest: 50,
	converters: {
		mpg: (value) => {
			if (value < 15) {
				return [1, 0, 0];
			} else if (value < 30) {
				return [0, 1, 0];
			} else {
				return [0, 0, 1];
			}

			// Encode the label values into three encoded sets, one for each possible value of label or category.
			// [ < 15, < 30, > 30 ]
			// ["low", "medium", "high"]
		},
	},
});

const logisticRegression = new LogisticRegression(features, _.flatMap(labels), {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
});

logisticRegression.train();

// logisticRegression.weights.print();

logisticRegression.test(testFeatures, _.flatMap(testLabels));

// logisticRegression.predict([[215, 440, 2.16]]).print();
