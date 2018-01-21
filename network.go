package main

type network interface {
	synapsesOps
	activation
	cost
	forward()
	backward()
	learn()
	recognize()
}

type Perceptron struct {
	synapses   denseSynapses
	activation sygmoid
	cost       quadratic
}

func (n *Perceptron) forward(set []float64) (output []float64, hiddenOut [][]float64) {
	var iSum, oSum float64

	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}

	iSum = sygmoid(iSum) // Activation of signal at a hidden layer
	lm := len(n.synapses)  // Count of neurons of a hidden layer apart from bias neuron

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
		output = append(output, sygmoid(oSum))
	}

	return
}
func (n *Perceptron) backward()  {}
func (n *Perceptron) recognize() {}
