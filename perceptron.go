package main

import "fmt"

type Perceptron struct {
	synapses   [][]float64
	activation activation
	cost       cost
}

func (n *Perceptron) forward(set []float64) (output []float64, hiddenOut [][]float64) {
	var iSum, oSum float64

	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}

	iSum = n.activation.activate(iSum) // Activation of signal at a hidden layer
	lm := len(n.synapses)              // Count of neurons of a hidden layer apart from bias neuron

	for i := range n.synapses[0] {
		var outLine []float64
		oSum = 0

		for j := range n.synapses {
			jIOut := n.synapses[j][i] * iSum
			oSum += jIOut
			outLine = append(outLine, jIOut)
		}

		hiddenOut = append(hiddenOut, outLine)
		// Apply a bias
		oSum += synapses[lm-1][i] // Bias doesn't use weights. Bias is a weight without a signal.
		output = append(output, n.activation.activate(oSum))
	}

	return
}

func (n *Perceptron) backward(out, labels []float64, hiddenOut [][]float64) {
	var cost, zk float64
	hiddenLen := len(n.synapses)

	exceptBiases := n.synapses[:hiddenLen]
	for i, ak := range out { // outputs of an out layer
		zk = 0                            // out layer k neuron input (sum of a hidden layer outputs)
		for _, aj := range hiddenOut[i] { // Weighted outputs of a hidden layer k neuron
			// Count k neuron of out layer input (sum output layer input value)
			zk += aj
		}
		// Count an error derivative using delta rule
		cost = n.cost.costDerivative(ak, labels[i]) * n.activation.actDerivative(zk)
		for k := range exceptBiases {
			// Multiply an error by output of an appropriate hidden neuron
			// Correct a synapse immediately (Stochastic gradient)
			// TODO: implement ability to learn in batches not ONLY stochastically
			n.synapses[k][i] += n.learningRate * cost * hiddenOut[i][k]
		}
		// Correct biases
		// The gradient of the cost function with respect to the bias for each neuron is simply its error signal!
		n.synapses[hiddenLen][i] += cost
	}
	//return synapses
}

func (n *Perceptron) Recognize(set [][]float64) (prediction []float, hiddenOut [][][]float64) {
	// Loop through a data set
	// Return recognition and hidden loop
	// Log cost to determine gradient
	var pred float64
	var hidd [][]float64

	for _, v := range set {
		pred, hidd := n.forward(v)
		prediction = append(prediction, pred)
		hiddenOut = append(hiddenOut, hidd)
	}
	return
}

func (n *Perceptron) Learn(set, labels, [][]float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	for i, v := range set {
		prediction, hiddenOut := forward(v, n.synapses)
		backward(prediction, labels[i], hiddenOut) // Adjust synapses in place.
		fmt.Println(quadraticCost(prediction, labels[i]))
	}
}

func NewPerceptron(activation activation, cost cost) network {
	return &Perceptron{
		newDenseSynapses(),
		activation,
		cost,
	}	
}

