// Supervised Learning:
// Supervised learning is the machine learning task of learning a function that maps an input to an output
// based on example input-output pairs. It infers a function from labeled training data consisting of a set
// of training examples. In supervised learning, each example is a pair consisting of an input object (typically
// a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm
// analyzes the training data and produces an inferred function, which can be used for mapping new examples.
// Supervised Learning uses labeled inputs(meaning the input has a corresponding output label) to train
// models and learn outputs.

// Unsupervised Learning:
// Unsupervised learning uses unlabeled data to learn about patterns in the data.
// Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets
// consisting of input data without labeled responses. The most common unsupervised learning method is
// "cluster analysis" (clustering algorithm), which is used for exploratory data analysis to find hidden
// patterns or grouping in data.
// Algorithm figures on it's own without supervision how to group the data based on the patterns it finds
// in the data. "It does not have a labeled output to learn from. We are not giving the algorithm the right answer
// or examples or right labels in advance to learn from". It is used for clustering, dimensionality
// reduction, and association rule learning.
// To summarize a clustering algorithm, which is a type of unsupervised learning algorithm, takes data without
// labels and tries to automatically group them into clusters.
// Whereas in supervised learning, the data comes with both inputs x and input labels y, in unsupervised learning,
// the data comes only with inputs x but not output labels y, and the algorithm has to find some structure or
// some pattern or something interesting in the data.
// Some other clustering algorithms:
// 1. Anomaly detection  (To find unusual data points or patterns in the data like fraud detection)
// 2. Dimensionality reduction (Compress data while keeping the important information)
// 3. Association rule learning (To find interesting relations between attributes in the data)

// Reinforcement Learning:
// Reinforcement learning is an area of machine learning concerned with how software agents ought to take
// actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning
// invloves agent learning in interactive environments based on rewards and punishments or penalties.

const outputs = [
	[10, 0.5, 16, 1],
	[200, 0.5, 16, 4],
	[350, 0.5, 16, 4],
	[600, 0.5, 16, 6],
	// { dropPosition: 400, bounciness: 0.5, ballSize: 16,       bucket: 4 }
	//   --independent variables (features) ---    ------- dependent variable (labl) --------
];

k = 4;

// K Nearest Neighbors Algorithm (knn)
// "Birds of a feather flock together"
// Steps:
// 1. Pick a value for k
// 2. Calculate the distance from the new point(prediction point or prediction data) to all other points
// 3. Sort the distances and pick the first k points
// 4. Count the number of points in each category among the k points
// 5. Return the category with the highest count
// This is a classification algorithm (not a regression algorithm)

// To implement a regression algorithm, we would have to take the average of the k points instead of
// counting the number of points in each category and returning the category with the highest count.

function knn(data, predictionPoint, k) {
	let result = data
		.map((row) => {
			return [distance(row.slice(0, row.length - 1), predictionPoint), row[row.length - 1]]; // [distance, bucket]
		})
		.sort((a, b) => a[0] - b[0]) // sort by distance in ascending order
		.slice(0, k) // take the first k elements
		.reduce((acc, next) => {
			acc[next[1]] ? (acc[next[1]] = acc[next[1]] + 1) : (acc[next[1]] = 1);
			return acc;
		}, {}); // { bucket: count }

	let countArray = []; // [[bucket, count]]
	for (let key in result) {
		countArray.push([key, result[key]]);
	}

	let bucket = countArray.sort((a, b) => b[1] - a[1])[0][0]; // sort by count in descending order // return the bucket with the highest count

	return parseInt(bucket);
}

// Takes only dropPosition into account (one independent variable)
// function distance(pointA, pointB) {
//     return Math.abs(pointA - pointB)
// }

// Finding distance for N independent variables (dropPosition, bounciness, ballSize)
// pointA = [dropPosition, bounciness, ballSize]
// pointB = [dropPosition, bounciness, ballSize]
function distance(pointA, pointB) {
	// zip pointA and pointB into a single array of arrays
	// [[dropPositionA, dropPositionB], [bouncinessA, bouncinessB], [ballSizeA, ballSizeB]]
	let zippedArray = pointA.map((row, index) => {
		return [row, pointB[index]];
	});

	// calculate the distance for each independent variable
	let distance = Math.sqrt(
		zippedArray.reduce((acc, next) => {
			return acc + Math.pow(next[0] - next[1], 2);
		}, 0)
	);

	return distance;
}

// split data into training and testing sets
function splitData(data, testCount) {
	// shuffle the data set so that the test set and training set are not biased or skewed
	// towards a particular category.
	let shuffled = data.slice(0);
	for (let i = shuffled.length - 1; i > 0; i--) {
		let randomIndex = Math.floor(Math.random() * (i + 1));
		let temp = shuffled[i];
		shuffled[i] = shuffled[randomIndex];
		shuffled[randomIndex] = temp;
	}

	let testSet = shuffled.slice(0, testCount);
	let trainingSet = shuffled.slice(testCount);

	return [testSet, trainingSet];
}

function runAnalysis() {
	let testSetSize = 10;
	// let [testSet, trainingSet] = splitData(outputs, testSetSize)
	let [testSet, trainingSet] = splitData(minMax(outputs, 3), testSetSize); // normalize the data

	let correct = 0;
	let incorrect = 0;

	testSet.forEach((row) => {
		let bucket = knn(trainingSet, row.slice(0, row.length - 1), k);
		if (bucket === row[row.length - 1]) {
			correct++;
		} else {
			incorrect++;
		}
	});

	console.log(`Accuracy: ${(correct / testSetSize) * 100}%`);
}

// Finding an optimal k value
function findOptimalK(data) {
	let testSetSize = 10;
	let [testSet, trainingSet] = splitData(minMax(outputs, 3), testSetSize); // normalize the data

	let kArray = Array(15).fill(0);

	kArray.forEach((row, index) => {
		let correct = 0;
		let incorrect = 0;
		let k = index + 1;
		testSet.forEach((row) => {
			let bucket = knn(trainingSet, row.slice(0, row.length - 1), k);
			if (bucket === row[row.length - 1]) {
				correct++;
			} else {
				incorrect++;
			}
		});

		console.log(`Accuracy for k = ${k}: ${(correct / testSetSize) * 100}%`);
	});
}

// Normalizing the data (Feature Normalization or Feature Scaling)
// Each independent variable (or feature) should have the same weight while calculating the distance
// If one feature has a much larger range of values than another, the distance will be dominated by
// that feature and the other features will be ignored (or have a much smaller impact on the distance or
// will have smaller contribution to the calculated distance.)
// This will result in a bad prediction model. That is why we need to normalize the data.
// Normalization(Scaling) is the process of transforming values of several variables of different
// scale into a similar range.

// For feature scaling, aim for about -1 <= xj <= 1 for each feature xj
// -3 <= xj <= 3, -0.3 <= xj <= 0.3 are also acceptable ranges

// Common normalization techniques:

// 1. Min-max scaling
//  In this technique, the values are shifted and rescaled so that they end up ranging from 0 to 1.
//  This can be treated kind of interpolation between the original values and the values with the
//  mean value. The formula is:
//  For each feature (independent variable) in each row of the data set:
//  normalizedValue = featureValue - minOfFeatureValues / maxOfFeatureValues - minOfFeatureValues

// 2. Mean normalization
//  In this technique, the values are rescaled so that they are centered around 0 mostly ranging from -1 to 1.
//  The formula is:
//  For each feature (independent variable) in each row of the data set:
//  normalizedValue = featureValue - meanOrAverageOfFeatureValues / maxOfFeatureValues - minOfFeatureValues

// 3. Z score normalization
//  In this technique, the values are rescaled so that they have a mean of 0 and a standard deviation of 1.
//  The formula is:
//  For each feature (independent variable) in each row of the data set:
//  normalizedValue = featureValue - meanOrAverageOfFeatureValues / standardDeviationOfFeatureValues

// 4. Dividing by the maximum value:
//  In this technique, the values are rescaled so that they are mostly ranging from 0 to 1.
//  The formula is:
//  For each feature (independent variable) in each row of the data set:
//  normalizedValue = featureValue / maxOfFeatureValues

function minMax(data, featureCount) {
	// featureCount = number of independent variables in the data set (dropPosition, bounciness, ballSize)

	let clonedData = [...data];

	// iterate over each column (one column for each independent variable at a time)
	for (let i = 0; i < featureCount; i++) {
		let column = clonedData.map((row) => row[i]); // get the column values for each row
		let min = Math.min(...column);
		let max = Math.max(...column);

		// iterate over each row and normalize the value for each independent variable
		clonedData.forEach((row) => {
			row[i] = (row[i] - min) / (max - min);
		});

		// OR
		// for (let j = 0; j < clonedData.length; j++) {
		//     clonedData[j][i] = (clonedData[j][i] - min) / (max - min)
		// }
	}

	return clonedData;
}

// Feature Selection:
// Choosing the right independent variables to include in our analysis to make predictions

// Selecting the right feature:
function findBestFeature(data) {
	// data = outputs
	let testSetSize = 10;
	let k = 4;

	// iterate over each feature (dropPosition, bounciness, ballSize)
	// iterating over all the different features we have rather than over different
	// values of k because now we want to find the best feature to use for our analysis
	for (let i = 0; i < 3; i++) {
		let eachFeatureToLabel = data.map((row) => [row[i], row[row.length - 1]]); // [[feature, label]]
		let [testSet, trainingSet] = splitData(minMax(eachFeatureToLabel, 3), testSetSize);
		let correct = 0;
		let incorrect = 0;

		testSet.forEach((row) => {
			let bucket = knn(trainingSet, row.slice(0, row.length - 1), k);
			if (bucket === row[row.length - 1]) {
				correct++;
			} else {
				incorrect++;
			}
		});

		console.log(`Accuracy for k = ${k}: ${(correct / testSetSize) * 100}%`);
	}
}

// KNN for Regression
function knnRegression(data, predictionPoint, k) {
	let result =
		data
			.map((row) => {
				return [distance(row.slice(0, row.length - 1), predictionPoint), row[row.length - 1]];
			})
			.sort((a, b) => a[0] - b[0]) // sort by distance in ascending order
			.slice(0, k) // take the first k elements
			.reduce((acc, next) => {
				return acc + next[1];
			}, 0) / k; // take the average of the k elements

	return result;
}

// 3. Standardization: (Z score normalization)
// Standardization is the process of transforming data into a standard scale.
// Standardization becomes beneficial when dealing with algorithms that are sensitive to the scale of
// the input features.
// Standardization is useful when your data has varying scales and the algorithm you are using does not make
// assumptions about the distribution of your data, such as k-nearest neighbors and artificial neural networks.
// Standardization assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to
// be true, but the technique is more effective if your attribute distribution is Gaussian.
// Prefer standardization over normalization when the distribution of your data is Gaussian that in laymans terms
// means that the data is normally distributed (normally distributed data is data that is symmetrically distributed)
// i.e when a lot of values are clustered around some average value and while there are some values that are way
// way off the scale large(outliers)
// We apply standardization like normalization, per feature (independent variable) or per column
// basis, i.e calcualte mean and standard deviation for each column and then apply the standardization formula
// to each value in the column.
// For example if we do a min-max analysis(normalization) on a data that is normally distributed,
// the data will be skewed to the left or right and the data will not be normally distributed anymore.
// So in this case we should use standardization instead of normalization.
// To calculate standardization, use the following:

// Z =  X - μ / σ
//  Where:
// Z is the standardized value,
// X is the original value,
// μ is the mean of the variable
// X, and σ is the standard deviation of the variable  X.

// μ = ∑X / n
// σ = √∑(X - μ)² / n
// n is the number of data points.

// Standardized Value = (value - mean or average) / standard deviation | (for each value in the data set)
// Standard deviation = square root of the variance
// Variance = sum of the squared differences between each value and the mean or average value

function standardize(data, featureCount) {
	// featureCount = number of independent variables in the data set (dropPosition, bounciness, ballSize)

	let clonedData = [...data];

	// iterate over each column (one column for each independent variable at a time)
	for (let i = 0; i < featureCount; i++) {
		let column = clonedData.map((row) => row[i]); // get the column values for each row
		let mean = column.reduce((acc, next) => acc + next, 0) / column.length; // mean of the column
		let variance = column.reduce((acc, next) => acc + Math.pow(next - mean, 2)) / column.length;
		let standardDeviation = Math.sqrt(variance);

		// iterate over each row and standardize the value for each independent variable (i.e each column)
		clonedData.forEach((row) => {
			row[i] = (row[i] - mean) / standardDeviation;
		});
	}

	return clonedData;
}
