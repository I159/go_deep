package go_deep

type Perceptron struct {
	activation
	cost
	learningRate float64
	batchSize    int
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
	lm := len(n.synapses) - 1          // Count of neurons of a hidden layer apart from bias neuron

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

func (n *Perceptron) backward(currLayerOut, labels []float64, prevLayerOut, correction [][]float64) [][]float64 {
	var cost, zk float64
	prevLayerSize := len(n.synapses) - 1

	for i, ak := range currLayerOut {
		zk = 0
		for _, aj := range prevLayerOut[i] {
			zk += aj // Sum current layer input
		}
		// Delta rule
		cost = n.cost.costDerivative(ak, labels[i]) * n.activation.actDerivative(zk)
		for k := 0; k < prevLayerSize; k++ {
			// Corrections vector of the same shape as synapses vector
			correction[k][i] += cost * prevLayerOut[i][k]
		}
		// Add bias correction
		correction[prevLayerSize][i] += cost
	}
	return correction
}

func (n *Perceptron) Learn(set, labels [][]float64) (costGradient []float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	var batchCost []float64

	prevLayerSize := len(n.synapses)
	currLayerSize := len(n.synapses[0])
	correction := make([][]float64, prevLayerSize)

	for i := range correction {
		correction[i] = make([]float64, currLayerSize)
	}

	for i, v := range set {
		if batchCounter >= n.batchSize {
			for j := 0; j < prevLayerSize; j++ {
				for k := 0; k < currLayerSize; k++ {
					n.synapses[j][k] += n.learningRate * correction[j][k] / float64(n.batchSize)
				}
			}

			batchCounter = 0
			costSum := 0.0
			correction := make([][]float64, prevLayerSize)
			for i := range correction {
				correction[i] = make([]float64, currLayerSize)
				costSum += batchCost[i]
			}
			costGradient = append(costGradient, costSum/float64(n.batchSize))
			batchCost = []float64{}
		}

		prediction, hiddenOut := n.forward(v, true)
		correction = n.backward(prediction, labels[i], hiddenOut, correction)
		batchCost = append(batchCost, n.cost.countCost(prediction, labels[i]))

		batchCounter++
	}
	return
}

func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64) {
	var pred []float64

	for _, v := range set {
		pred, _ = n.forward(v, false)
		prediction = append(prediction, pred)
	}
	return
}

func (n *Perceptron) Measure(set, labels [][]float64) (float64, []float64) {
	var pred []float64
	var cost []float64
	accuracy := map[bool]float64{true: 0, false: 0}

	for i, v := range set {
		pred, _ = n.forward(v, false)
		cost = append(cost, n.countCost(pred, labels[i]))

		maxPred := 0.
		maxPredIdx := 0
		for j, r := range pred {
			if r > maxPred {
				maxPred = r
				maxPredIdx = j
			}
		}
		for k, v := range labels[i] {
			if v == 1 {
				accuracy[k == maxPredIdx]++
			}
		}
	}
	return accuracy[true] / accuracy[false], cost
}

func NewPerceptron(
	learningRate float64,
	activation activation,
	cost cost,
	input,
	hidden,
	output float64,
	batchSize int) network {

	return &Perceptron{
		activation:   activation,
		cost:         cost,
		learningRate: learningRate,
		batchSize:    batchSize,
		synapses:     newDenseSynapses(hidden, input, output),
	}
}
