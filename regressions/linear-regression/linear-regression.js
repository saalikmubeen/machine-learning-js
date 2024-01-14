const tf = require("@tensorflow/tfjs");

// Linear Regression is used for predicting a continuous value like the price of a house or the miles per gallon
// of a car. It is not used for predicting a discrete value like whether a car is a sedan or a truck.

// Methods of solving linear regression:
// 1. Ordinary Least Squares
// 2. Gereralized Least Squares
// 3. Maximum Likelihood Estimation
// 4. Bayesian Regression
// 5. Gradient Descent

// Mean Squared Error
// Mean squared Error tells us how close a regression line is to a set of points. In laymans terms,
// it tells us how accurate or bad the guess was from the actual value. The smaller the number, the
// better the guess or better the regression line fits the data.
// MSE = 1/n * ∑(y - y_hat)^2 | Mean Squared Error = 1/n * ∑(Guessi - Actuali)^2

// MSE is unlikely to ever be 0, but the closer it is to 0, the better the model is at predicting.

// y = mx + b
// Miles Per Gallon (mpg) = m * (Horsepower) + b

class LinearRegression {
	constructor(features, labels, options) {
		this.features = features; // independent variables like horsepower, weight, displacement
		this.labels = labels; //  corresponding dependent variables like mpg
		this.options = Object.assign({ learningRate: 0.1, iterations: 100, batchSize: 50 }, options);
		this.m = 0;
		this.b = 0;

		// this.featuresTensor = tf.tensor(this.features);
		this.featuresTensor = this.processFeatures(features);
		this.labelsTensor = tf.tensor(this.labels);
		// this.weights = tf.zeros([2, 1]);

		this.weights = tf.zeros([this.featuresTensor.shape[1], 1]); // [2, 1]

		this.mseHistory = [];
	}

	// iterations = max number of times we want to run our gradient descent algorithm

	train() {
		// Gradient Descent
		// for (let i = 0; i < this.options.iterations; i++) {
		// 	this.gradientDescent(this.featuresTensor, this.labelsTensor);
		// 	this.recordMSE();
		// 	this.updateLearningRate();
		// }

		// Batch Gradient Descent
		const batchQuantity = Math.floor(this.featuresTensor.shape[0] / this.options.batchSize);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const { batchSize } = this.options;
				const startIndex = j * batchSize;
				const featureSlice = this.featuresTensor.slice([startIndex, 0], [batchSize, -1]);
				const labelSlice = this.labelsTensor.slice([startIndex, 0], [batchSize, -1]);
				this.gradientDescent(featureSlice, labelSlice);
			}
			this.recordMSE();
			this.updateLearningRate();
		}

		// Stochastic Gradient Descent
		// Same as Batch Gradient Descent, just with batchSize = 1
	}

	// Without using Tensorflow library | Plain JS
	// Univariate Linear Regression
	gradientDescent2() {
		// Steps:
		// 1. Guess values for m and b
		// 2. Calculate the MSE
		// 3. Calculate the slope of MSE with respect to m and b
		// 4. Multiply the slope by the learning rate
		// 5. Subtract the result of above step from the current value of m and b
		// 6. Repeat steps 2-5 until the MSE is close to 0

		// y = mx + b
		// Mean Squared Error = 1/n * ∑(Guessi - Actuali)^2
		// Guessi = guessed_m * x + guessed_b

		// Slope of MSE with respect to b and m:
		// ∂MSE/∂b = 2/n * ∑(guessed_m * x + guessed_b - Actuali)
		// ∂MSE/∂m = 2/n * ∑(x * (guessed_m * x + guessed_b - Actuali))

		// mpg = m * (Horsepower) + b

		const currentGuessesForMPG = this.features.map((row) => {
			let x = row[0]; // Horsepower
			return this.m * x + this.b; // guessed_m * x + guessed_b
		});

		let n = this.features.length; // number of rows

		const mseSlopeB =
			currentGuessesForMPG
				.map((guess, idx) => {
					return guess - this.labels[idx][0]; // (guessed_m * x + guessed_b) - Actuali
				})
				.reduce((acc, next) => acc + next, 0) *
			(2 / n);

		const mseSlopeM =
			currentGuessesForMPG
				.map((guess, idx) => {
					return this.features[idx][0] * (guess - this.labels[idx][0]); // x * ((guessed_m * x + guessed_b) - Actuali)
				})
				.reduce((acc, next) => acc + next, 0) *
			(2 / n);

		this.b = this.b - mseSlopeB * this.options.learningRate;
		this.m = this.m - mseSlopeM * this.options.learningRate;
	}

	// Vectorized Solution of Gradient Descent
	// Multivariate Linear Regression (Works for any number of features or independent variables)
	// Using Tensorflow library
	gradientDescent(featuresTensor, labelsTensor) {
		// featuresTensor  = [n, 2]

		// Matrix multiplication [n, 2] * [2, 1] = [n, 1
		let currentGuessesForMPG = featuresTensor.matMul(this.weights); // Matrix multiplication

		const differences = currentGuessesForMPG.sub(labelsTensor); // guess - actual | [n, 1]

		const slopes = featuresTensor // [n, 2]
			.transpose() // [2, n]
			.matMul(differences) // [2, n] * [n, 1] = [2, 1]
			.div(featuresTensor.shape[0])
			.mul(2); // [2, 1] * 2/n = [2, 1] | [[mseSlopeM], [mseSlopeB]]

		// MatrixMultiply = ***
		// slopes = 2/n * [Transpose(featuresTensor)] *** [(featuresTesnor *** weights) - labelsTensor]

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
	}

	// coefficientOfDetermination
	test(testFeatures, testLabels) {
		testFeatures = this.processFeatures(testFeatures);
		testLabels = tf.tensor(testLabels);

		let predictions = testFeatures.matMul(this.weights);

		let averageOfLabels = testLabels.mean();

		let ssTotal = testLabels.sub(averageOfLabels).pow(2).sum().arraySync();
		let ssResidual = testLabels.sub(predictions).pow(2).sum().arraySync();

		let coefficientOfDetermination = 1 - ssResidual / ssTotal; // r2

		console.log("Coefficient of Determination is:", coefficientOfDetermination);
		return coefficientOfDetermination;
	}

	standardize(features) {
		if (!this.mean || !this.variance) {
			const { mean, variance } = tf.moments(features, 0); // 0 means column wise mean and variance
			this.mean = mean;
			this.variance = variance;
		}

		return features.sub(this.mean).div(this.variance.pow(0.5));
	}

	processFeatures(features) {
		features = tf.tensor(features);

		// Standardization of each feature
		features = this.standardize(features);

		// Add a column of 1s to the features tensor to make it [n, 2]
		features = features.concat(tf.ones([features.shape[0], 1]), 1);

		return features;
	}

	// Custom Learning Rate Optimizer
	// Some of the popular Learning Rate Optimization methods are:
	// 1. Adagrad, 2. RMSProp, 3. Adam, 4. Momentum, 5. Nesterov Momentum, 6. AdaDelta, 7. AdaMax, 8. Nadam
	// We will implement our own custom learning rate optimizer:
	// Steps:
	// 1. With every iteration of gradient descent, calculate the exact value of MSE and store it.
	// 2. After running an iteration of gradient descent, look or compare the current MSE with the previous MSE.
	// 3. If the MSE went "up", then we did a bad update i.e we overshot the minimum and we need to reduce
	// the learning rate. So we will reduce the learning rate by 50%, divide it by 2.
	// 4. If the MSE went "down", then we did a good update i.e we are going in the right direction and we can
	// increase the learning rate by 5%. So we will increase the learning rate by 5%, multiply it by 1.05.
	// 5. Repeat steps 1-4 until the MSE is close to 0.

	recordMSE() {
		// Mean Squared Error = 1/n * ∑(Guessi - Actuali)^2 | Guessi = guessed_m * x + guessed_b

		let mse = this.featuresTensor
			.matMul(this.weights)
			.sub(this.labelsTensor)
			.pow(2)
			.sum()
			.div(this.featuresTensor.shape[0])
			.arraySync();

		this.mseHistory.unshift(mse); // Add the mse to the beginning of the mseHistory array
	}

	updateLearningRate() {
		if (this.mseHistory.length < 2) {
			return;
		}

		if (this.mseHistory[0] > this.mseHistory[1]) {
			// MSE went up
			this.options.learningRate = this.options.learningRate / 2;
		} else {
			this.options.learningRate = this.options.learningRate * 1.05;
		}
	}

	predict(observations) {
		return this.processFeatures(observations).matMul(this.weights);
	}

	print() {
		// console.log("Updated M is:", this.m, "Updated B is:", this.b);

		console.log(
			"Updated M is:",
			this.weights.arraySync()[0][0],
			"Updated B is:",
			this.weights.arraySync()[1][0]
		);
	}
}

module.exports = LinearRegression;

// Gradient Descent Alterations:
// 1. Batch Gradient Descent
// 2. Stochastic Gradient Descent

// In Gradient Descent, we use the entire dataset to calculate the gradient or slope of the MSE cost function.

// Batch Gradient Descent:
// In Batch Gradient Descent, we use a portion of the observations in our feature set to calculate the gradient
// or slope of the MSE and our current guess for m and b. We then use that gradient obtained from the portion
// of the observations to update our guess for m and b. And so we refer to this as batch gradient descent
// because we are taking batches of our observation set and then running gradient descent with that.
// With Batch Gradient Descent, in theory we're going to get some convergence on our optimal values of M
// and B slightly faster than with normal gradient descent because we are updating M and B more frequently.

// Stochastic Gradient Descent (SGD):
// In Stochastic Gradient Descent, we take a single observation from our feature set and we use that single
// observation at a time to calculate the gradient or slope of the MSE. We then use that gradient obtained
// from that single observation to update our guess for m and b.
// Stochastic gradient descent is the same as batch gradient descent, just with a batch size of one.
// So if we batch everything out to one observation, we end up getting the same thing as stochastic gradient
// descent as well.

// Why we do this?

// Well, again, at the end of the day, we end up getting some convergence on our values of M and B much
// faster than if we attempt to loop through our entire data set before we update M and B just one time.
// With batch and stochastic gradient descent, we're going to be updating M and B constantly and constantly and
// constantly as we are iterating through our data set.
// And so we might end up finding optimal values of M and B after only one, two, 3 or 5 iterations using
// stochastic gradient descent.
// And we might end up getting some optimal value of M and B with batch gradient descent and only 10 or
// 15 iterations.
// And so it can significantly speed up the time that it takes to train our model and get some idea of
// how we can relate all of our different vehicle attributes to the miles per gallon.
// With batch and stochastic gradient descent, we're still going to iterate over the entire data set.
// So we're still looking at all the records. The only difference here is how often we are updating the
// values of M and B
