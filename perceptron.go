package main

type Perceptron struct {
	activation
	cost
	learningRate float64
	synapses     [][]float64
}

func (n *Perceptron) forward(set []float64, keepHidden bool) (output []float64, hiddenOut [][]float64) {
	var iSum, oSum float64

	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}

	iSum = n.activation.activate(iSum) // Activation of signal at a hidden layer
	lm := len(n.synapses) - 1               // Count of neurons of a hidden layer apart from bias neuron 

	for i := range n.synapses[0] {
		var outRaw []float64
		oSum = 0

		for j := range n.synapses {
			jIOut := n.synapses[j][i] * iSum
			oSum += jIOut
			if keepHidden {
				outRaw = append(outRaw, jIOut)
			}
		}

		if keepHidden {
			hiddenOut = append(hiddenOut, outRaw)
		}
		// Apply a bias
		oSum += n.synapses[lm][i] // Bias doesn't use weights. Bias is a weight without a signal.
		output = append(output, n.activation.activate(oSum))
	}

	return
}

func (n *Perceptron) backward(out, labels []float64, hiddenOut [][]float64) {
	var cost, zk float64
	hiddenLen := len(n.synapses) - 1

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

func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64) {
	var pred []float64

	for _, v := range set {
		pred, _ = n.forward(v, false)
		prediction = append(prediction, pred)
	}
	return
}


func (n *Perceptron) Learn(set, labels [][]float64) (costGradient []float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	for i, v := range set {
		prediction, hiddenOut := n.forward(v, true)
		n.backward(prediction, labels[i], hiddenOut) // Adjust synapses in place in stochastic manner
		costGradient = append(costGradient, n.cost.countCost(prediction, labels[i]))
	}
	return
}

func NewPerceptron(learningRate float64, activation activation, cost cost, input, hidden, output float64) network {
	return &Perceptron{
		activation,
		cost,
		learningRate,
		newDenseSynapses(hidden, input, output),
	}
}
