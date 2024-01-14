const fs = require("fs");

// ** Without using lodash

function isNumber(value) {
	return typeof value == "number";
}

function _isNaN(value) {
	// An `NaN` primitive is the only value that is not equal to itself.
	// Perform the `toStringTag` check first to avoid errors with some
	// ActiveX objects in IE.
	return isNumber(value) && value != +value;
}

function shuffleData(data) {
	const shuffled = [...data];
	for (let i = 0; i < shuffled.length; i++) {
		let randomIndex = Math.floor(Math.random() * (i + 1)); // random index from 0 to i
		// swap
		let temp = shuffled[i];
		shuffled[i] = shuffled[randomIndex];
		shuffled[randomIndex] = temp;
	}
	return shuffled;
}

function extractColumns(data, columnNames) {
	const headers = data[0];

	const colIndexes = columnNames.map((column) => headers.indexOf(column));
	const extracted = data.map((row) => row.filter((_, index) => colIndexes.includes(index)));

	return extracted;
}

function loadCSV(
	filename,
	{ dataColumns = [], labelColumns = [], converters = {}, shuffle = false, splitTest = false }
) {
	let data = fs.readFileSync(filename, { encoding: "utf-8" });
	data = data.split("\n").map((d) => d.split(","));
	data = data.filter((val) => !val.every((v) => v === ""));
	const headers = data[0];

	data = data.map((row, index) => {
		if (index === 0) {
			// headers row
			return row;
		}
		return row.map((element, index) => {
			if (converters[headers[index]]) {
				const converted = converters[headers[index]](element);
				return _isNaN(converted) ? element : converted;
			}

			const result = parseFloat(element.replace('"', ""));
			return isNaN(result) ? element : result;
		});
	});

	if (shuffle) {
		let headers = data[0];
		let dataWithoutHeaders = data.slice(1, data.length);
		dataWithoutHeaders = shuffleData(dataWithoutHeaders);
		data = [headers, ...dataWithoutHeaders];
	}

	let labels = extractColumns(data, labelColumns);
	data = extractColumns(data, dataColumns);

	data.shift(); // remove headers
	labels.shift(); // remove headers

	if (splitTest) {
		const trainSize = typeof splitTest === "number" ? splitTest : Math.floor(data.length / 2);

		return {
			testFeatures: data.slice(0, trainSize),
			testLabels: labels.slice(0, trainSize),
			features: data.slice(trainSize),
			labels: labels.slice(trainSize),
		};
	} else {
		return {
			features: data,
			labels: labels,
			testFeatures: [],
			testLabels: [],
		};
	}
}

module.exports = loadCSV;
