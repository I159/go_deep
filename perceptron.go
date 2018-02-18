package go_deep

type Perceptron struct {
	input       inputLayer
	hiddenFirst hiddenFirstLayer
	hidden      []hiddenLayer
	output      outputLayer
	//activation
	//cost
	//learningRate float64
	//synapses     [][]float64
}

// TODO: deprecated.
//func (n *Perceptron) inputLayer(set []float64) (output [][]float64) {
	//// Each neuron of a first hidden layer receives all signals from input layer
	//// and sums it. Input layer doesn't change input signal
	//var iSum float64

	//for _, i := range set {
		//iSum += i
	//}

	//iSum *= .00001

	//nextLayerSize := len(n.synapses[0])
	//for i := range n.synapses {
		//output = append(output, make([]float64, nextLayerSize))
		//for _ = range n.synapses[i] {
			//output[i] = append(output[i], iSum)
		//}
	//}
	//return
//}

//func (n *Perceptron) hiddenForward(rowInput [][]float64) ([][]float64, [][]float64) {
//var iSum float64
//var input []float64
//currLayerSize := len(n.synapses)
//output := make([][]float64, len(n.synapses[0]))

//for _, raw := range rowInput {
//for _, item := range raw {
//iSum += item
//}
//input = append(input, n.activate(iSum))
//}

//// Transition between layers is a matrix reshape. Way or another reshape matrix is required on step of multiplication or sum.
//var j int
//for i := range n.synapses[0] {
//for j = 0; j < currLayerSize-1; j++ {
//if output[i] == nil {
//output[i] = make([]float64, currLayerSize)
//}
//output[i][j] = n.synapses[j][i] * input[j]
//}
//output[i][j+1] += n.synapses[j+1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
//}

//// Keep layer input for backward propagation
//return output, rowInput
//}

//func (n *Perceptron) outputForward(rowInput [][]float64) ([]float64, [][]float64) {
//var output []float64
//var iSum float64

//for _, raw := range rowInput {
//iSum = 0
//for _, item := range raw {
//iSum += item
//}
//output = append(output, n.activate(iSum))
//}

//// Keep a layer input for backward propagation
//return output, rowInput
//}

func (n *Perceptron) backward(prediction [][]float64, labels []float64) {
	n.hiddenFirst.backward(
		n.output.backward(labels)
	)
	//var cost, zk float64
	//prevLayerSize := len(n.synapses) - 1

	//for i, ak := range currLayerOut {
		//zk = 0
		//for _, aj := range prevLayerOut[i] {
			//zk += aj // Sum current layer input
		//}
		//// Delta rule
		//cost = n.cost.costDerivative(ak, labels[i]) * n.activation.actDerivative(zk)
		//for k := 0; k < prevLayerSize; k++ {
			//// Corrections vector of the same shape as synapses vector
			//correction[k][i] += cost * prevLayerOut[i][k]
		//}
		//// Add bias correction
		//correction[prevLayerSize][i] += cost
	//}
	//return correction
}

func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) (costGradient [][]float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	costGradient = make([][]float64, n.layersCount)

	for j := 0; j <= epochs; j++ {
		for i, v := range set {
			if batchCounter >= batchSize {
				n.hiddenFirst.applyCorrections()
				//for j := 0; j < prevLayerSize; j++ {
					//for k := 0; k < currLayerSize; k++ {
						//n.synapses[j][k] += n.learningRate * correction[j][k] / float64(batchSize)
					//}
				}

				batchCounter = 0
				//costSum := 0.0
				//correction := make([][]float64, prevLayerSize)
				//for i := range correction {
					//correction[i] = make([]float64, currLayerSize)
					//costSum += batchCost[i]
				//}
				//costGradient = append(costGradient, costSum/float64(n.batchSize))
				//batchCost = []float64{}
			}

			prediction, costs := n.forwardMeasure(v, labels[i])
			for k, cost := range costs {
				costGradient[k] = append(costGradient[k], cost)
			}
			n.backward(prediction, labels[i])
			// TODO: compute global cost of the network, possibly per layer
			//prediction, hiddenOut := n.forward(v)

			//correction = n.backward(prediction, labels[i], hiddenOut, correction)
			//batchCost = append(batchCost, n.cost.countCost(prediction, labels[i]))

			batchCounter++
		}
	}
	return
}

func (n *Perceptron) forward(rowInput []float64) []float64 {
	// NOTE: this is a single layer implementation
	return n.output.forward(
		n.hiddenFirst.forward(
			n.input.forward(rowInput),
		),
	)
}

func (n *Perceptron) forwardMeasure(rowInput, labels []float64) (prediction []float64, cost float64) {
	// Get per layer prediction and cost with layers `forwardMeasure` methods
	return
}


func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64) {
	var pred []float64

	for _, v := range set {
		pred, _ = n.forward(v)
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
