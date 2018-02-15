package go_deep

type Perceptron struct {
	activation
	cost
	learningRate float64
	batchSize    int
	epochs       int
	synapses     [][]float64
}

func (n *Perceptron) inputLayer(set []float64) (activatedSums []float64) {
	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	var iSum float64

	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}

	iSum = n.activation.activate(iSum) // Activation of signal at a hidden layer
	for _ = range n.synapses {
		activatedSums = append(activatedSums, iSum)
	}

	return
}

func (n *Perceptron) forward(input []float64) (output []float64, midOut [][]float64) {
	var inputSum int
	currLayerSize := len(n.synapses)
	midOut = make([][]float64, len(n.synapses[0]))

	for i := range n.synapses[0] {

		oSum = 0
		for j := 0; j < currLayerSize - 1; j++ {
			if midOut[i] == nil {
				midOut[i] == make([]float64, currLayerSize)
			}
			midOut[i][j] = n.synapses[j][i] * input[j]
		}
		midOut[i][j+1] += n.synapses[j+1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
	}

	// Sum and activate output/input of a next layer
	for _, raw := range midOut {
		inputSum = 0
		for _, item := range raw {
			inputSum += n.activate(item)
		}
		output = append(inputSum)
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

func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) (costGradient []float64) {
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

	for j := 0; j <= epochs; j++ {
		for i, v := range set {
			if batchCounter >= batchSize {
				for j := 0; j < prevLayerSize; j++ {
					for k := 0; k < currLayerSize; k++ {
						n.synapses[j][k] += n.learningRate * correction[j][k] / float64(batchSize)
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

func NewPerceptron(
	learningRate float64,
	activation activation,
	cost cost,
	input,
	hidden,
	output float64) network {

	return &Perceptron{
		activation:   activation,
		cost:         cost,
		learningRate: learningRate,
		synapses:     newDenseSynapses(hidden, input, output),
	}
}
