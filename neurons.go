package go_deep

type inputLayer interface {
	synapseInitializer
	forward([]float64) [][]float64
	backward([][]float64)
	applyCorrections(float64)
}

type hiddenLayer interface {
	Activation
	synapseInitializer
	forward([][]float64) [][]float64
	backward([][]float64) [][]float64
	applyCorrections(float64)
}

type outputLayer interface {
	Activation
	cost
	forwardMeasure([][]float64, []float64) ([]float64, float64)
	forward(rowInput [][]float64) []float64
	backward(prediction, labels []float64) [][]float64
}

type inputDense struct {
	synapseInitializer
	corrections, synapses        [][]float64
	nextLayerSize, currLayerSize int
	learningRate                 float64
}

func (l *inputDense) forward(input []float64) (output [][]float64) {
	for i := 0; i < l.nextLayerSize; i++ {
		for j, v := range input {
			if output[i] == nil {
				output[i] = make([]float64, l.currLayerSize)
			}
			output[i] = append(output[i], l.synapses[j][i]*v)
		}
	}
	return
}

func (l *inputDense) backward(eRRors [][]float64) {
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i, eRR := range eRRors {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}

		for j, c := range eRR {
			l.corrections[i][j] += c
		}
	}
}

func (l *inputDense) applyCorrections(batchSize float64) {
	for i, corr := range l.corrections {
		for j, c := range corr {
			l.synapses[i][j] += l.learningRate * c / batchSize
		}
	}
}

func NewInputDense(curr, next int, bias, learningRate float64) inputLayer {
	layer := &inputDense{
		Activation: activation,
		synapseInitializer: denseSynapses{},
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  learningRate,
	}
	// There is no previous layer but incoming data is flat it means that
	// input signal for a neuron of an input layer is not a sum but a single value
	layer.synapses = layer.init(1, curr, next, bias)
	return layer
}

type hiddenDense struct {
	Activation
	synapseInitializer
	prevLayerSize, currLayerSize, nextLayerSize int
	learningRate                                float64
	corrections, fromSynapses, toSynapses       [][]float64 // TODO: synapses directed from a previous layer to the current one used at back propagation
	// to compute error for correction of the "to" synapses
}

func (l *hiddenDense) forward(input [][]float64) (output [][]float64) {
	var activated []float64
	var inputSum float64
	output = make([][]float64, l.nextLayerSize)

	for _, i := range input {
		inputSum = 0
		for _, j := range i {
			inputSum += j
		}
		activated = append(activated, l.activate(inputSum))
	}

	for i := 0; i < l.nextLayerSize; i++ {
		for j, v := range activated {
			if output[i] == nil {
				output[i] = make([]float64, l.currLayerSize)
			}
			// Transition between layers is a matrix reshape. Way or another reshape matrix is required on step of multiplication or sum.
			output[i][j] = l.fromSynapses[j][i] * v
		}
		output[i][l.currLayerSize-1] = l.synapses[l.currLayerSize-1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
	}

	return output
}

func (l *hiddenDense) backward(eRRors [][]float64) {
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i, eRR := range eRRors {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}

		for j, c := range eRR {
			// TODO: implement complete backprop as follows https://theclevermachine.wordpress.com/2014/09/06/derivation-error-backpropagation-gradient-descent-for-neural-networks/
			l.corrections[i][j] += c
		}
	}
}

func (l *hiddenDense) applyCorrections(batchSize float64) {
	for i, corr := range l.corrections {
		for j, c := range corr {
			l.synapses[i][j] += l.learningRate * c / batchSize
		}
	}
}

func newHiddenDense(prev, curr, next int, bias, learningRate float64, activation Activation) firstHiddenLayer {
	layer := &hiddenDense{
		Activation: activation,
		synapseInitializer: &denseSynapses{},
		prevLayerSize: prev,
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  learningRate,
	}
	layer.synapses = layer.init(prev, curr, next, bias)
	return layer
}

type outputDense struct {
	Activation
	// Cost function exists only in output layer and in hidden layers used indirectly
	// as a sum of weighted errors. Thus cost function is global for a network.
	cost
	prevLayerSize int
	input         [][]float64
}

func (l *outputDense) forward(rowInput [][]float64) (output []float64) {
	l.input = rowInput
	var iSum float64

	for _, raw := range rowInput {
		iSum = 0
		for _, item := range raw {
			iSum += item
		}
		output = append(output, l.activate(iSum))
	}
	return
}

func (l *outputDense) forwardMeasure(rowInput [][]float64, labels []float64) (prediction []float64, cost float64) {
	prediction = l.forward(rowInput)
	cost = l.countCost(prediction, labels)
	return
}

func (l *outputDense) backward(prediction []float64, labels []float64) (corrections [][]float64) {
	var cost, zk float64
	corrections = make([][]float64, l.prevLayerSize)

	for i, ak := range prediction {
		zk = 0
		for _, aj := range l.input[i] {
			zk += aj // Sum current layer input
		}
		// Delta rule
		cost = l.costDerivative(ak, labels[i]) * l.actDerivative(zk)
		for k := 0; k < l.prevLayerSize-1; k++ {
			// Corrections vector of the same shape as synapses vector
			corrections[k] = append(corrections[k], cost*l.input[i][k])
		}
		// Add bias correction
		corrections[l.prevLayerSize-1] = append(corrections[l.prevLayerSize-1], cost)
	}
	return
}

func newOutput(prev, curr int, activation Activation, cost cost) outputLayer {
	return &outputDense{
		Activation:    activation,
		cost:          cost,
		prevLayerSize: prev,
	}
}
