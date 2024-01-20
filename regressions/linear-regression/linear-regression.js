const tf = require("@tensorflow/tfjs");

// Linear Regression is used for predicting a continuous value like the price of a house or the miles per gallon
// of a car. It is not used for predicting a discrete value like whether a car is a sedan or a truck.

// y = mx + b (this is the equation of a line and the equation of a linear regression model)
// m and b are the "parameters of the model" or "weights of the model" or "coefficients of the model"
// and are the variables you can adjust during training in order to improve the model.

// Methods of solving linear regression:
// 1. Ordinary Least Squares
// 2. Generalized Least Squares
// 3. Maximum Likelihood Estimation
// 4. Bayesian Regression
// 5. Gradient Descent

// Mean Squared Error (Cost Function or Loss Function):
// The Cost function will tell us how well the model is doing so that we can try to get it to do better.
// It is a measure of how well the line fits the training data.
// The cost function takes the prediction ŷ and compares it to the target y by taking ŷ - y.
// This difference is called the error.

// Mean squared Error (squared error cost function) tells us how close a regression line is to a set
// of points. In laymans terms, it tells us how accurate or bad the guess was from the actual value.
// The smaller the number, the better the guess or better the regression line fits the data.
// J(m, b) = MSE = 1/2n * ∑(y - ŷ)^2 | Mean Squared Error = 1/n * ∑(Guessi - Actuali)^2
//  ŷ (y_hat) = "predicted value" or estimated of y or the output of our model
//  y = "target value" or the actual value of y

// Guessi = guessed_m * x + guessed_b

// MSE is unlikely to ever be 0, but the closer it is to 0, the better the model is at predicting.

/*
You want to fit a straight line to the training data, so you have the model, f(w, b)(x) =  wx + b. 
Here, the model's parameters are w, and b. Now, depending on the values chosen for these parameters, you get 
different straight lines. You want to find values for w, and b, so that the straight line fits the training 
data well. To measure how well a choice of w, and b fits the training data, you have a cost function J. What 
the cost function J does is, it measures the difference between the model's predictions(ŷ), and the actual 
true values for y. The linear regression would try to find values for w, and b, that make J(w, b) as small 
as possible, we want to minimize the cost function J(w, b)


What you really want is an efficient algorithm that you can write in code for automatically finding the values 
of parameters w and b they give you the best fit line. That minimizes the cost function j. 
There is an algorithm for doing this called "gradient descent". Gradient descent and variations on gradient 
descent are used to train, not just linear regression, but some of the biggest and most 
complex models in all of AI.

*/

/*

Model           ->       y = f(w, b)(x) = wx + b
Parameters      ->       w, b
Cost Function   ->       J(w, b) = 1/2n * ∑(y(i) - ŷ(i))^2
Goal            ->       Minimize J(w, b)

*/

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
		// 1. Guess initial values for m and b say 0 and 0
		// 2. Calculate the MSE
		// 3. Calculate the slope of MSE with respect to m and b
		// 4. Multiply the slope by the learning rate, α
		// 5. Subtract the result of above step from the current value of m and b
		// 6. Repeat steps 2-5 until the MSE is close to 0 and our algorithm has converged

		// y = mx + b
		// Mean Squared Error = 1/n * ∑(Guessi - Actuali)^2
		// Guessi = guessed_m * x + guessed_b

		// Slope of MSE with respect to b and m:
		// ∂MSE/∂b = 2/n * ∑(guessed_m * x + guessed_b - Actuali)
		// ∂MSE/∂m = 2/n * ∑(x * (guessed_m * x + guessed_b - Actuali))

		// ∂MSE/∂b = 2/n * ∑(Guessi - Actuali)
		// ∂MSE/∂m = 2/n * ∑(x * (Guessi - Actuali))
		//    where Guessi = guessed_m * x + guessed_b

		// updated_w = w - α * ∂MSE/∂w
		// updated_b = b - α * ∂MSE/∂b

		// If slope is positive, then we are too far to the right and we need to decrease the value of b
		// i.e subtract the slope from the current value of b
		// If slope is negative, then we are too far to the left and we need to increase the value of b
		// i.e add the slope to the current value of b
		// That's why there is a negative sign in the formula for updated_b

		// If α is too small, gradient descent can be slow to converge.
		// If α is too large, gradient descent can overshoot the minimum. It may even fail to converge
		// and will never find the minimum.

		// As we get nearer a minimum, gradient descent will automatically take smaller steps.
		// And that is because the slope of the MSE with respect to m and b will get smaller as we get
		// nearer a minimum and that means the update steps will get smaller as well.

		// The gradient descent algorithm can be used to minimize any cost function J, not just the MSE
		// cost function that we are using here for linear regression.

		// Putting together gradient descent with the MSE cost function gives us the "linear regression algorithm".

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
	// Multiple Linear Regression (Works for any number of features or independent variables)
	// y = m1x1 + m2x2 + m3x3 + ... + b
	// xi = represents the featrues of the ith training example, (ith row of the features matrix,
	// so it's going to be a list of vector that includes all the features for the ith training example)
	// xj = represents the jth feature (of the ith training example, ith row of the features matrix)

	//  (i)
	// x    -> value of feature j in the ith training example (x superscript (i) subscript j)
	//  j

	// f(w, b) = w1x1 + w2x2 + w3x3 + ... + b
	// w = [w1, w2, w3, ...] -> vector of weights
	// x = [x1, x2, x3, ...] -> vector of features

	// f(w, b) = w . x + b (dot product of w and x)

	// Vectoization is the process of rewriting an algorithm so that it uses matrix operations instead of
	// for loops.

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
// So at each iteration or step of gradient descent, we are using all the training exapmles or rows
// all at a time to calculate the gradient or slope of the MSE cost function.

// Batch Gradient Descent:
// In Batch Gradient Descent, we use portions or batches of the observations in our feature set to
// calculate the gradient or slope of the MSE and our current guess for m and b.
// We then use that gradient obtained from the smaller portions of the observations to update our guess for
// m and b. And so we refer to this as batch gradient descent because we are taking batches of our observation
// set and then running gradient descent with that.
// With Batch Gradient Descent, in theory we're going to get some convergence on our optimal values of M
// and B slightly faster than with normal gradient descent because we are updating M and B more frequently.

// Stochastic Gradient Descent (SGD):
// Stochastic gradient descent is the same as batch gradient descent, just with a batch size of one.
// So if we batch everything out to one observation, we end up getting the same thing as stochastic gradient
// descent as well. We call the algorithm stochastic gradient descent because the gradient or slope of the
// MSE cost function used to update coefficients or weights are noisy or not exact or stochastic.

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
// ** With batch and stochastic gradient descent, we're still going to iterate over the entire data set.
// ** So we're still looking at all the records. The only difference here is how often we are updating the
// ** values of M and B. For example if we are iterating over the entire data set 100 times, with batch
// ** size of 10, we will update the values of M and B 10 times for each iteration. So we will update the
// ** values of M and B 1000 times in total.

/*
I want to make a quick aside or a quick side note on an alternative way for finding w and b for linear regression. 
This method is called the "normal equation". Whereas it turns out gradient descent is a great method for 
minimizing the cost function J to find w and b, there is one other algorithm that works only for linear regression 
and pretty much none of the other algorithms for solving for w and b and this other method does not need an iterative gradient descent algorithm. Called the "normal equation method", it turns out to be possible to use an advanced linear algebra library to just solve for w and b all in one goal without iterations. Some disadvantages of the normal equation method are; first unlike gradient descent, this is not generalized to other learning algorithms, such as the logistic regression algorithm. The normal equation method is also quite slow if the number of features and this large.
*/
