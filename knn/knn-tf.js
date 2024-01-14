require("@tensorflow/tfjs-node"); // to run the calculations the CPU
const tf = require("@tensorflow/tfjs");

const loadCSV = require("./load-csv");

// 2D Array of Lat and Long
// let features = tf.tensor([
// 	[121, -47],
// 	[121.2, 46.6],
// 	[-122, 46.4],
// 	[-120.9, 46.7],
// ]);

// Price correspnding to [lat, long] in the features array.
// let labels = tf.tensor([[200], [250], [215], [240]]);

// ley predictionPoint = tf.tensor([-121.5, 47]);

function knn(data, predictionPoint, k) {
	const { mean, variance } = tf.moments(features, 0); // 0 means column wise mean and variance

	// Use same mean and variance to apply Standardization to the predictionPoint as well as to the data
	let standardizedPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

	return (
		data
			.sub(mean)
			.div(variance.pow(0.5)) // Standardization of each feature
			.sub(standardizedPrediction)
			.pow(2)
			.sum(1)
			.sqrt()
			.expandDims(1)
			.concat(labels, 1) // [[distance1, label1], [distance2, label2], ...]
			.unstack()
			// unstack the tensor into an array of tensors where each row is now a tensor stuffed into a normal
			// javascript array. So unstacking creates a normal array containing tensors).
			.sort((a, b) => a.arraySync()[0] - b.arraySync()[0]) // sorted javascript array of tensors
			.slice(0, k)
			.reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
	);
}

console.log(knn(features, pp, 2));

let { features, labels, testFeatures, testLabels } = loadCSV("house-data.csv", {
	shuffle: true,
	splitTest: 10, // 10 rows for testing
	dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
	labelColumns: ["price"],
});

features = tf.tensor(features);
labels = tf.tensor(labels);

let k = 10;

testFeatures.forEach((testPoint, i) => {
	const result = knn(features, labels, tf.tensor(testPoint), k);
	const err = (testLabels[i][0] - result) / testLabels[i][0];
	console.log("Error", err * 100);
});
