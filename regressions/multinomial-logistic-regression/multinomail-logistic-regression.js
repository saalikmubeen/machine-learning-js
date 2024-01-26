const tf = require("@tensorflow/tfjs");

/*

Multinomail Logistic Regression:
With multinomail have the ability to apply multiple different label values to classify a given observation.
For example, we could classify a given observation as either "red", "green", or "blue", more than two possible values.
Another example is, given a "person's age" (observation or feature or independent variable), we want to predict 
whether that person prefers "reading books" or "watching movies" or "dancing" (label or dependent variable).

Sample Data set and its encoding:

Age         Preference            Encoded Preference           Encoded Preference       Encoded Preference
(feature)    (Label)               for "reading books"          for "watching movies"    for "dancing"

18         reading books              1                             0                        0
15         watching movies            0                             1                        0
25         dancing                    0                             0                        1
30         dancing                    0                             0                        1
35         reading books              1                             0                        0
40         watching movies            0                             1                        0
45         reading books              1                             0                        0

In multinomail logistic regression, we encode each of those different label values, just as we did before
in logistic regression, as either a 0 or a 1. But instead of just doing that encoding step one time, 
we're going to do it three separate times, one for each of the different possible label values.
So if possible label values were 5, we will create 5 different sets of encododed values.

We're going to take a look at all the different possible classification values in here.
Before we had just two. Now we have three.
So I would look through all my different label values and I would notice that there are three distinct
possible label values watch movies, read books and dance. I would then use each of those three different 
label values to produce a new, different encoded label set. 

So now I've got these three different label value columns.
Now I'm going to take those three different label value columns and I'm going to pass each of them to three
separate logistic regression classes.So in other words, I can make three instances of our class.
I'll take my entire feature set. And then first the first label set, then the second label set, and then the
third label set. So I'm going to have three different instances of this class.
I'll then train each of those instances separetely for each label values set. And so I would essentially 
have three copies of logistic regression that is really good for making predictions about i) whether or not someone wants to watch movies or not, ii) someone wants to dance iii) someone wants to read books.

Remember, the zeros indicates we do not want to do or we do not want to classify something as whatever
the one label is.

So the first class or instance of our LogisticRegression class ould be really good at identifying 
people who want to watch movies, the second class would really good at identifying people who want to 
read books. And then finally, the third one would be really good at predicting whether someone wants
to dance or not.


So let's say if we now want to figure out what activity a 20 year old prefers, we would essentially
toss that 20 value into each of our three different models, and each one would spit out a probability
of whether or not that person is going to want to watch movies, read books or dance.
So let's say that for someone who's 20 years old, there's a 0.34 probability of them wanting to watch
movies a 0.47 probability of one of them wanting to read books and a zero probability of them wanting
to dance. And so in this case, we would look over these three different probabilities coming out of the three
different models, and we would select which whichever probability was highest in this case, the highest
probability is reading books. And so we would say that the 20 year old person wants to read books.

Now, if we start to think about the different values that are held inside of each of these different
class instances, we can kind of break it out like this.
So for logistic regression instance number one (#1), that is essentially responsible for predicting whether
or not someone likes to watch movies, we would have a couple of different instance variables.
First off, we would have a unique set of weights (vlaues of M and B)
Remember, the weights tensor holds our different values of M and B that make our model or basis function
fit the data. We would also have a unique set of labels for each instance as well.
So as we just said a second ago, each of these different labels tensors would end up being slightly
different. And then in addition, we would of course have our features tensor in each one.
Now, the first thing I want to point out here is that the features tensor inside of each instance of
logistic regression would end up being completely identical.

This is the basic idea of multinomail LogisticRegression.

Little smarter way of doing the same thing:
Instead of making three different instances of our class, we can just make one instance of our class
and concat all of our different label values together into one big label tensor and concat all of our
different weights together into one big weights tensor such that each column in big labels tensor
represents the labels for different possible classification value.



Marginal vs Conditional Probability Distribution:
Marginal Probability Distribution considers one possible output case in isolation.
Conditional Probability Distribution considers all possible output cases together when putting 
together a probability.

Marginal Probability Distribution -> 0.34 probability of just "watching movies"

Conditional Probability Distribution -> 0.34 probability of just "watching movies" and NOT NOT WANTING TO
READ BOOKS OR DANCE. 1/6 chance of rolling a 1 and not rolling a 2,3,4,5,6

Sigmoid function calculates the "Marginal Probability Distribution".

Softmax function calculates the "Conditional Probability Distribution".


Softmax function: σ(z)=  e^z / ∑(e^z)
Softmax function = (e^ mx + b) / ∑(e^ mx + b)
Softmax gives the "probability of being the one label rather than 0 label".

For example if z = {1, 2, 3, 4, 5} 
then σ(z) = {0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865}

each value in σ(z) can be interpretted as the probability of being that value 
in z rather than any other value in z. for example, 0.63640865 is the probability of being 
5 and not 1,2,3,4. 
The sum of all values in σ(z) must be 1.

In case of multinomail logistic regression we want to use Softmax as basis function and not 
sigmoid.


*/

class LogisticRegression {
	constructor(features, labels, options) {
		this.features = this.processFeatures(features);
		this.labels = tf.tensor(labels);
		this.crossEntropyHistory = []; // costHistory

		this.options = Object.assign({ learningRate: 0.1, iterations: 100, batchSize: 50 }, options);

		// weights has as many columns as the number of columns in the labels tensor
		this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
	}

	gradientDescent(features, labels) {
		this.weights = tf.tidy(() => {
			const currentGuesses = features.matMul(this.weights).softmax();
			const differences = currentGuesses.sub(labels);

			const slopes = features.transpose().matMul(differences).div(features.shape[0]);

			// slopes = 1/n * [Transpose(featuresMatrix)] *** [(softmax(featuresMatrix *** weights)) - labelsTensor]

			return this.weights.sub(slopes.mul(this.options.learningRate));

			// this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
		});
	}

	train() {
		// Batch Gradient Descent
		const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const { batchSize } = this.options;
				const startIndex = j * batchSize;
				const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
				const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
				this.gradientDescent(featureSlice, labelSlice);
			}
			this.recordCrossEntropy();
			this.updateLearningRate();
		}
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

			const filler = variance.cast("bool").logicalNot().cast("float32");

			this.mean = mean;
			this.variance = variance.add(filler);
			// to avoid divide by zero error when standardizing with zero variance
		}

		return features.sub(this.mean).div(this.variance.pow(0.5));
	}

	predict(observations) {
		// observations = x;
		// prediction = 1 / (1 + e^-(optimal_m * x + optimal_b))
		let prediction = this.processFeatures(observations).matMul(this.weights).softmax();

		// argMax returns the column index of the maximum or largest value inside each
		// row of the tensor
		return prediction.argMax(1); // 1 means row wise argMax
	}

	test(testFeatures, testLabels) {
		// And so remember that in theory, the largest value in each row represents the highest
		// probability of some particular label needing to be applied to that particular observation.

		const predictions = this.predict(testFeatures);
		testLabels = tf.tensor(testLabels).argMax(1);

		const incorrect = predictions.notEqual(testLabels).sum().arraySync();

		let accuracy = (predictions.shape[0] - incorrect) / predictions.shape[0];
		console.log("Accuracy:", accuracy);
		return accuracy;
	}

	recordCrossEntropy() {
		// Vectorized Matrix Form of the Cross Entropy Equation:
		// Cross Entropy(CE) = - 1 / n * Transpose(ActualMatrix) ***
		//             log(GuessesMatix) + Transpose(1 - ActualMatrix) *** log(1 - GuessesMatrix)
		let cost = tf.tidy(() => {
			let guessedValues = this.features.matMul(this.weights).softmax();

			let firstTerm = this.labels.transpose().matMul(guessedValues.log());
			let secondTerm = this.labels.mul(-1).add(1).transpose().matMul(
				guessedValues
					.mul(-1)
					.add(1)
					.add(1e-7) // arbitary change to avoid log(0) which is -Infinity
					.log()
			);

			let cost = firstTerm.add(secondTerm).div(this.features.shape[0]).mul(-1).arraySync()[0][0];

			return cost;
		});

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
