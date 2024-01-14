const tf = require("@tensorflow/tfjs");

/*
Linear Regression is used to predict the continuous values like price, salary, age, etc.
Logistic Regression is used to predict the discrete values like 0 or 1, true or false, etc
In other words, Linear Regression is used to solve Regression problems whereas
Logistic Regression is used to solve Classification problems.

Binary Classification:  (Natural Binary Classification)
Binary Classification is essentially when we try to take an observation and then put it
into one or of two categories, like 0 or 1, true or false, yes or no, etc
For example, given a "person's age" (observation or feature or independent variable), we want to predict 
whether that person prefers "reading books" or "watching movies" (label or dependent variable).

Basic Logistic Regression to do Binary Classification:

Sigmoid Function:
Sigmoid function is a mathematical function having a characteristic "S"-shaped curve or
sigmoid curve. Often, sigmoid function refers to the special case of the logistic function
Sigmoid: 1 / (1 + e^-x)
e id Euler's number (2.71828)

The Sigmoid Function always returns a value between 0 and 1 and never will go beyond that
range. So, if we pass a value of x to the sigmoid function, it will always return a value
between 0 and 1. That is why, we use the sigmoid function in Logistic Regression to predict
the probability of an observation belonging to a particular class or category
(discrete values like 0 or 1, true or false, etc). 
Assuming we have only label values of 0 and 1, sigmoid gives the "probability of being the 
'1' label".

y =  1 / (1 + e^-(mx + b))

Logistic Regression Gradient Descent:
Calculating or finding the best values for m and b in the above equation is the main goal
of the Logistic Regression Gradient Descent algorithm. We use the Gradient Descent algorithm
to find the best values for m and b in the above equation. It's similar to the Linear
Regression Gradient Descent algorithm except that we use the Sigmoid Function in Logistic
Regression instead of the Linear Equation in Linear Regression.
Steps:
1. Encode the label values (y) as either 0 or 1 ("One Hot Encoding")
2. Guess the starting values for b and m (m1, m2, m3, ...)
3. Calculate the slope of MSE with respect to b and m (m1, m2, m3, ...) using all observations
   in the feature set (training set) and current m and b values
4. Multiply the slope by the learning rate
5. Subtract the result from the current m and b values and assign the new values to m and b
6. Repeat steps 2, 3 and 4 until the slope is very close to 0

In Linear Regression, MSE is used as our metric of how badly we guessed our values of M and B.
In Logistic Regression, we use Cross Entropy as our metric of how badly we guessed our values
of M and B. Cross Entropy is a measure of how different our prediction is from the actual
value of y. Cross Entropy is also known as Log Loss.
Cross Entropy Equation: -(y * log(p) + (1 - y) * log(1 - p))
Cross Entropy(CE) = - 1 / n * ∑Actual * log(Guess) + (1 - Actual) * log(1 - Guess)
Guess in case of Logistic Regression is the Sigmoid Function (1 / (1 + e^-(mx + b)))

Vectorized Matrix Form of the Cross Entropy Equation:
Cross Entropy(CE) = - 1 / n * Transpose(ActualMatrix) ***
            log(GuessesMatix) + Transpose(1 - ActualMatrix) *** log(1 - GuessesMatrix)

Actual Means the actual values of y (labels) in the training set and Guesses means the
predicted values of y (labels) in the training set calculated using the Sigmoid Function
(1 / (1 + e^-(guessed_m * x + guessed_b)))

So it turns out through sheer luck, through sheer chance that the derivative of this entropy equation
above with respect to M and B, when we put it into that vectorized matrix form, ends up being identical
to the derivative of the MSE equation with respect to M and B in Linear Regression in the vectorized
matrix form.
∂MSE/∂b,m = // ∂CE/∂b,m

Matrix Multiplication = ***
slopes = 1/n * [Transpose(featuresMatrix)] *** [(sigmoid(featuresMatrix *** weights)) - labelsMatrix]

With logistic regression, when we're using this sigmoid equation, this r squared equation (coefficient  of
Determination) value doesn't make any sense anymore.
So, we need a new metric to measure how well our model is doing at the end after all the training
and calculating optimal values of m and b.
And that metric is called Accuracy.
Accuracy = Number of Correct Predictions / Total Number of Predictions
Accuracy = Total Number of Predictions - Number of Incorrect Predictions / Total Number of Predictions

In many blog posts and in ML literature, you will see the term "cost function" used instead for
"MSE" and "Cross Entropy". In this course, we've been referring them to functions that tell us how badly we
are guessing the values of M and B, which is what they really, truly are.
But in a lot of scientific documentation, they are referred to as cost functions.
And the "sigmoid" and "linear equation" are referred to as "hypothesis function" or "basis function".

Decision Boundary:
Decision Boundary is the line that separates the two classes. In other words, it's the line that
separates the two classes in the feature space. In our case, for passedemissions, the decision
boundary is the line that separates the cars that passed the emissions test from the cars that
failed the emissions test. If we set a threshold of 0.5 as our decision boundary, then all the
cars that have a probability of passing the emissions test greater than 0.5 (in other words whose predicted
or guessed value of y is greater than 0.5) will be classified as passed the emissions test and
will be assigned a value of 1. And all the cars that have a probability of passing the emissions test
less than 0.5 will be classified as failed the emissions test as will be assigned a value of 0.


1. Encoding the label values (y) as either 0 or 1:
   "One Hot Encoding"

Age     Preference            Encoded "Preference"   
18      reading books              1                      
15      watching movies            0                             
25      reading books              1                             
30      watching movies            0                             
35      reading books              1 
40      watching movies            0 

*/

class LogisticRegression {
	constructor(features, labels, options) {
		this.features = this.processFeatures(features);
		this.labels = tf.tensor(labels);
		this.crossEntropyHistory = []; // costHistory

		this.options = Object.assign(
			{ learningRate: 0.1, iterations: 100, decisionBoundary: 0.5, batchSize: 50 },
			options
		);

		this.weights = tf.zeros([this.features.shape[1], 1]);
	}

	gradientDescent(features, labels) {
		const currentGuesses = features.matMul(this.weights).sigmoid();
		const differences = currentGuesses.sub(labels);

		const slopes = features.transpose().matMul(differences).div(features.shape[0]);

		// slopes = 1/n * [Transpose(featuresMatrix)] *** [(sigmoid(featuresMatrix *** weights)) - labelsTensor]

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
	}

	train() {
		// Gradient Descent
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent(this.features, this.labels);
			this.recordCrossEntropy();
			this.updateLearningRate();
		}

		// Batch Gradient Descent
		// const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
		// for (let i = 0; i < this.options.iterations; i++) {
		// 	for (let j = 0; j < batchQuantity; j++) {
		// 		const { batchSize } = this.options;
		// 		const startIndex = j * batchSize;
		// 		const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
		// 		const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
		// 		this.gradientDescent(featureSlice, labelSlice);
		// 	}
		// 	this.recordCrossEntropy();
		// 	this.updateLearningRate();
		// }
	}

	processFeatures(features) {
		features = tf.tensor(features);

		// Standardization of each feature
		features = this.standardize(features);

		// Add a column of 1s to the features tensor to make it [n, 2]
		features = features.concat(tf.ones([features.shape[0], 1]), 1);

		return features;
	}

	standardize(features) {
		if (!this.mean || !this.variance) {
			const { mean, variance } = tf.moments(features, 0); // 0 means column wise mean and variance
			this.mean = mean;
			this.variance = variance;
		}

		return features.sub(this.mean).div(this.variance.pow(0.5));
	}

	test(testFeatures, testLabels) {
		const predictions = this.predict(testFeatures);
		testLabels = tf.tensor(testLabels);

		// if the difference between the predicted value and the actual value is 0 then it means
		// that the prediction was correct otherwise it was incorrect
		const incorrect = predictions.sub(testLabels).abs().sum().arraySync();

		// Accuracy = Total Number of Predictions - Number of Incorrect Predictions / Total Number of Predictions
		let accuracy = (predictions.shape[0] - incorrect) / predictions.shape[0];
		console.log("Accuracy:", accuracy);
		return accuracy;
	}

	predict(observations) {
		// observations = x;
		// prediction = 1 / (1 + e^-(optimal_m * x + optimal_b))
		let prediction = this.processFeatures(observations).matMul(this.weights).sigmoid();

		// If the probability of passing the emissions test is greater than 0.5, then we classify
		// the car as passed the emissions test (1) otherwise we classify the car as failed the
		// emissions test (0)
		return prediction.greater(this.options.decisionBoundary).cast("float32");
	}

	recordCrossEntropy() {
		// Vectorized Matrix Form of the Cross Entropy Equation:
		// Cross Entropy(CE) = - 1 / n * Transpose(ActualMatrix) ***
		//             log(GuessesMatix) + Transpose(1 - ActualMatrix) *** log(1 - GuessesMatrix)
		let guessedValues = this.features.matMul(this.weights).sigmoid();

		let firstTerm = this.labels.transpose().matMul(guessedValues.log());
		let secondTerm = this.labels.mul(-1).add(1).transpose().matMul(guessedValues.mul(-1).add(1).log());

		let cost = firstTerm.add(secondTerm).div(this.features.shape[0]).mul(-1).arraySync()[0][0];

		this.crossEntropyHistory.unshift(cost);
	}

	updateLearningRate() {
		if (this.crossEntropyHistory.length < 2) {
			return;
		}

		if (this.crossEntropyHistory[0] > this.crossEntropyHistory[1]) {
			this.options.learningRate /= 2;
		} else {
			this.options.learningRate *= 1.05;
		}
	}
}

module.exports = LogisticRegression;

/*
It turns out that we still could use MSE as a metric to decide how to update both our
learning rate. And at the end of the day, even if the derivatives between mean squared error and the 
cross-entropy equation were different (in reality they are smae), we could still use MSE as our 
basis for gradient descent as well if we needed to.
The reason that we don't use MSE for this logistic regression stuff is that when we use
the MSE function with specifically the sigmoid function, the relationship between our
guesses to "b" and MSE go from that nice convex parabola having a global minima to a kind of
wavy function that has a bunch of local minimums.

When we say convex function, that means that it has one minimum value, So we know that whenever we look at 
that slope value and it tells us to increase or decrease 'b' we know
that we are always headed towards the global minimum of that line or that equation and that's always
going to give us the optimal value of 'b' at that point. But when we use the sigmoid function, we
end up with this wavy function that has a bunch of local minimums. And so when we use gradient descent
with this wavy function, we might accidentally fall into one of those local minimums and not the global
minimum. And so that's why we don't use mean squared error with logistic regression. We use this cross
entropy equation instead.

So specifically because we are introducing this sigmoid, the shape of the relationship between the
value of 'b' and MSE is going to change into this kind of non convex or wavy function.


And so essentially using this non convex function or this wave function might accidentally lead us to
a local minimum that is not the truly global minimum or the most optimal value of 'b'.
So we might get tricked into thinking that this is a value right here and that, hey, this is
probably the best value of 'b'.
So that's why we do not choose to use the MSE metric to get kind of a history or a value
of how good or bad our guess of "b" and "m" are.

So in other words, we can kind of sometimes luck out using MSE and still get the right answer.
Sometimes we're not going to get this wavy function and fall into this local minimum right here.
Sometimes we're going to luck out and still get into the global minimum.
But to make sure that this works correctly 100% of the time, we use that cross entropy function instead.

So, to make sure that that never happens, then we would want to use cross entropy equation instead, 
which is going to guarantee that we're not going to have any of those little local minimums.

When we use the cross entropy equation, we instead get a fully convex function, just as we did previously
with MSE back on linear regression.
*/
