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


For logistic regression, f(x) is the sigmoid (logistic) function, whereas for linear regression, 
f(x) is a linear function.

Sigmoid Function:
Sigmoid function is a mathematical function having a characteristic "S"-shaped curve or
sigmoid curve. Often, sigmoid function or logistic function refers to the special case of the logistic function
Sigmoid: 1 / (1 + e^-x)
e id Euler's number (2.71828)

The Sigmoid Function always returns a value between 0 and 1 and never will go beyond that
range. So, if we pass a value of x to the sigmoid function, it will always return a value
between 0 and 1. That is why, we use the sigmoid function in Logistic Regression to predict
the probability of an observation belonging to a particular class or category
(discrete values like 0 or 1, true or false, etc). 
Assuming we have only label values of 0 and 1, sigmoid gives the "probability of being the 
'1' label". 
Interpretation of the logistic regression output -> "Probability that the class is 1"
The way I encourage you to think of logistic regressions output is to think of it as outputting
the probability that the class or the label y will be equal to 1 given a certain input x. 

y =  1 / (1 + e^-(mx + b))

σ(z) = 1 / (1 + e^-z)
z = mx + b

f(m, b)(x) = logistic regression model = σ(z) = 1 / (1 + e^-(mx + b)) = P(y = 1 | x; m, b)
P(y = 1 | x; m, b) = Probability that y is 1 given the input x and the parameters m and b
P(y=1) + P(y=0) = 1

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

Remember that the cost function gives you a way to measure how well a specific set of parameters 
fits the training data. For linear regression, we used the MSE cost function.
In Linear Regression, MSE is used as our metric of how badly we guessed our values of M and B.
In Logistic Regression, we use Cross Entropy as our metric of how badly we guessed our values
of M and B. Cross Entropy is a measure of how different our prediction is from the actual
value of y. Cross Entropy is also known as Log Loss.
Cross Entropy Equation: -(y * log(p) + (1 - y) * log(1 - p))
Cross Entropy(CE) = - 1 / n * ∑Actual * log(Guess) + (1 - Actual) * log(1 - Guess)
Guess in case of Logistic Regression is the Sigmoid Function (1 / (1 + e^-(mx + b)))

To expand the actual equation, we have:

Loss = {
		- log(p) if y = 1
		- log(1 - p) if y = 0
}

This particular cost function is derived from statistics using a statistical principle called 
"maximum likelihood estimation", which is an idea from statistics on how to efficiently find 
parameters for different models. This cost function has the nice property that it is convex.


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

∂MSE/∂b = 2/n * ∑(Guessi - Actuali)
∂MSE/∂m = 2/n * ∑(x * (Guessi - Actuali))
       where Guessi = 1 / (1 + e^-(guessed_m * x + guessed_b))

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

Decision Boundary: (Threshold)
Decision Boundary is the line that separates the two classes. In other words, it's the line that
separates the two classes in the feature space. In our case, for passedemissions, the decision
boundary is the line that separates the cars that passed the emissions test from the cars that
failed the emissions test. If we set a threshold of 0.5 as our decision boundary, then all the
cars that have a probability of passing the emissions test greater than 0.5 (in other words whose predicted
or guessed value of y is greater than 0.5) will be classified as passed the emissions test and
will be assigned a value of 1. And all the cars that have a probability of passing the emissions test
less than 0.5 will be classified as failed the emissions test as will be assigned a value of 0.

If our decision boundary or threshold is 0.5, i.e if the predicted value of y is greater than 0.5,
then we classify the observation as 1, otherwise we classify the observation as 0,
then f(m, b)(x) will be > 0.5 if σ(z) > 0.5
σ(z) > 0.5 if z > 0 i.e if mx + b > 0
In other words mx + b = 0 is the decision boundary or threshold line that separates the two classes in
the feature space if we set the threshold to 0.5.

Non Linear Decision Boundary:
If we have a non linear decision boundary, then we can't use a linear equation to find the
decision boundary. We have to use a non linear equation to find the decision boundary.
For example if our model is f(m, b)(x) = σ(w1x1^2 + w2x2^2 + b) where x1 and x2 are the two features
and w1 and w2 are the weights, then the decision boundary will be a circle for a threshold of 0.5.
i.e for a threshold of 0.5, to classify an observation as 1, the predicted value of y should be
greater than 0.5. So z should be greater than 0. So w1x1^2 + w2x2^2 + b should be greater than 0.
Therefore w1x1^2 + w2x2^2 + b = 0 is the decision boundary or threshold line that separates the two classes.




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

// ** Overfitting and Underfitting **

/*

1. Underfitting
say y = mx + b is underfitting the data, i.e it is not fitting the data very well.

The algorithm that does not fit the training data very well. The technical term for this is the model 
is underfitting the training data. Another term is the algorithm has "high bias".
Another way to think of this form of bias is as if the learning algorithm has a very strong preconception, 
or we say a very strong bias, say that the housing prices are going to be a completely linear function 
of the size despite data to the contrary. This preconception that the data is (say) linear causes it to 
fit a straight line that fits the data poorly, leading it to underfitted data


2. Overfitting
say y = m1x + m2x^2 + m3x^3 + + m4x^4 + b is overfitting the data, i.e it is fitting the data too well.

Overfitting occurs when the line or model fits the training set very well, it has fit the data almost too well, 
hence is overfit. It does not look like this model will generalize to new examples that's never seen before,
i.e it will not perform well on the test set. It will perform well on the training set but not on the test set.
Another term for this is that the algorithm has "high variance". In machine learning, many people will use the terms over-fit and high-variance almost interchangeably.
The intuition behind overfitting or high-variance is that the algorithm is trying very hard to fit every 
single training example. It turns out that if your training set were just even a little bit different, 
say one house was priced just a little bit more or little bit less, then the function that the algorithm 
fits could end up being totally different. If two different machine learning engineers were to fit the model, 
to just slightly different datasets, they could end up with totally different predictions or highly variable predictions. That's why we say the algorithm has high variance.


3. Good Fit (Generalization)
Say y = m1x + m2x^2 + b fits the data better, i.e it is fitting the data very well.	
The idea that you want your learning algorithm to do well, even on examples that are not on the training 
set, that's called "generalization". Technically we say that you want your learning algorithm to 
generalize well, which means to make good predictions even on brand new examples that it has never seen before


You can say that the goal machine learning is to find a model that hopefully is neither 
underfitting nor overfitting. In other words, hopefully, a model that has neither high bias 
nor high variance.

To recap, if you have too many features like the fourth-order polynomial on the right, then the 
model may fit the training set well, but almost too well or overfit and have high variance. 
On the flip side if you have too few features, it underfits and has high bias. 
In the example, using quadratic features x and x squared, that seems to be just right.

*/

/*

* Addressing Overfitting Issues:

1. One way to address this problem is to collect more training data, that's one option. If you're able to get 
more data, that is more training examples on sizes and prices of houses, then with the larger training set, 
the learning algorithm will learn to fit a function that is less wiggly


2. A second option for addressing overfitting is to see if you can use fewer features.
Choosing the most appropriate set of features to use is sometimes also called feature selection.
So, if you have a lot of features, you can choose to use only a subset of those features, to avoid overfitting.
The disadvantage of this approach is that you may be throwing away some information and useful features 
could be lost .

3. Now, this takes us to the third option for reducing overfitting called "regularization". Regularization works 
well when we have a lot of features, each of which contributes a bit to predicting y. In this case, 
regularization will tend to reduce overfitting by driving the weights down to lower values. 

If you look at an overfit model, here's a model using polynomial features:
y = 28x + 285x^2 + 39x^3 - 174x^4 + b

You find that the parameters are often relatively large. Now if you were to eliminate some of these features, 
say, if you were to eliminate the feature x4, that corresponds to setting this parameter to 0. 
So setting a parameter to 0 is equivalent to eliminating a feature, which is what we saw in the 2nd method i.e 
feature selection. It turns out that regularization is a way to more gently reduce the impacts of some of the 
features without doing something as harsh as eliminating it outright. What regularization does is encourage 
the learning algorithm to shrink the values of the parameters without necessarily demanding that the parameter 
is set to exactly 0. It turns out that even if you fit a higher order polynomial like this, so long as you can 
get the algorithm to use smaller parameter values: w1, w2, w3, w4. You end up with a curve that ends up 
fitting the training data much better. So what regularization does, is it lets you keep all of your features, 
but they just prevents the features from having an overly large effect, which is what sometimes can 
cause overfitting. This helps you reduce overfitting when you have a lot of features and a relatively 
small training set.
By the way, by convention, we normally just reduce the size of the wj parameters, that is w1 through wn.
It doesn't make a huge difference whether you regularize the parameter b as well, you could do so if you want 
or not if you don't.

*/
