/*
We simply need to calculate the backpropagated error signal that reaches that layer \delta_l
and weight it by the feed-forward signal a_{l-1}feeding into that layer!
*/
package go_deep

type inputLayer interface {
	synapseInitializer
	forward([]float64) [][]float64
	backward([]float64)
	applyCorrections(float64)
}

type hiddenLayer interface {
	Activation
	synapseInitializer
	forward([][]float64) [][]float64
	backward([]float64) []float64
	applyCorrections(float64)
}

type outputLayer interface {
	Activation
	cost
	forwardMeasure([][]float64, []float64) ([]float64, float64)
	forward(rowInput [][]float64) []float64
	backward(prediction, labels []float64) []float64
}

type inputDense struct {
	synapseInitializer
	corrections, synapses        [][]float64
	nextLayerSize, currLayerSize int
	learningRate                 float64
	input                        []float64
}

func (l *inputDense) forward(input []float64) (output [][]float64) {
	l.input = input

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

func (l *inputDense) backward(eRRors []float64) {
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i, a := range l.input {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}
		for j, eRR := range eRRors {
			l.corrections[i][j] += eRR * a
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
		synapseInitializer: denseSynapses{},
		currLayerSize:      curr,
		nextLayerSize:      next,
		learningRate:       learningRate,
	}
	// NOTE: There is no previous layer but incoming data is flat it means that
	// input signal for a neuron of an input layer is not a sum but a single value
	// TODO: kind of bad design. Kept there to have synapses as pure slices. Possibly should be refactored... Later...
	layer.synapses = layer.init(1, curr, next, bias)
	return layer
}

type hiddenDense struct {
	Activation
	synapseInitializer
	prevLayerSize, currLayerSize, nextLayerSize int
	learningRate                                float64
	output, corrections, synapses               [][]float64
	input                                       []float64
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
		l.input = append(l.input, inputSum)
		activated = append(activated, l.activate(inputSum))
	}

	for i := 0; i < l.nextLayerSize; i++ {
		for j, v := range activated {
			if output[i] == nil {
				output[i] = make([]float64, l.currLayerSize)
			}
			// Transition between layers is a matrix reshape. Way or another reshape matrix is required on step of multiplication or sum.
			output[i][j] = l.synapses[j][i] * v
		}
		output[i][l.currLayerSize-1] = l.synapses[l.currLayerSize-1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
	}

	l.output = output
	return output
}

// FIXME: errors from an output layer are needed to compute actual corrections at hidden layers.
func (l *hiddenDense) backward(eRRors []float64) (nextLayerErrors []float64) {
	// Collect corrections for further forward error propagation
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i := 0; i < l.currLayerSize; i++ {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}
		for j := 0; j < l.nextLayerSize; j++ {
			// TODO: check matrix reshape correctness
			if i < l.currLayerSize-1 {
				l.corrections[i][j] += eRRors[j] * l.output[j][i]
			} else {
				// Bias
				l.corrections[i][j] += eRRors[j]
			}
		}
	}

	// Propagate backward
	var weightedErrSum float64
	for i, v := range l.input {
		acDer := l.actDerivative(v)
		weightedErrSum = 0

		for j, k := range l.synapses[i] {
			weightedErrSum += eRRors[j] * k
		}
		nextLayerErrors = append(nextLayerErrors, acDer*weightedErrSum)
	}

	return
}

func (l *hiddenDense) applyCorrections(batchSize float64) {
	for i, corr := range l.corrections {
		for j, c := range corr {
			l.synapses[i][j] += l.learningRate * c / batchSize
		}
	}
}

func newHiddenDense(prev, curr, next int, bias, learningRate float64, activation Activation) hiddenLayer {
	layer := &hiddenDense{
		Activation:         activation,
		synapseInitializer: &denseSynapses{},
		prevLayerSize:      prev,
		currLayerSize:      curr,
		nextLayerSize:      next,
		learningRate:       -learningRate,
	}
	// TODO: kind of bad design. Kept there to have synapses as pure slices. Possibly should be refactored... Later...
	layer.synapses = layer.init(prev, curr, next, bias)
	return layer
}

type outputDense struct {
	Activation
	// Cost function exists only in output layer and in hidden layers used indirectly
	// as a sum of weighted errors. Thus cost function is global for a network.
	input []float64
	cost
	prevLayerSize int
}

func (l *outputDense) forward(rowInput [][]float64) (output []float64) {
	var iSum float64

	for _, raw := range rowInput {
		iSum = 0
		for _, item := range raw {
			iSum += item
		}

		l.input = append(l.input, iSum)
		output = append(output, l.activate(iSum))
	}
	return
}

func (l *outputDense) forwardMeasure(rowInput [][]float64, labels []float64) (prediction []float64, cost float64) {
	prediction = l.forward(rowInput)
	cost = l.countCost(prediction, labels)
	return
}

func (l *outputDense) backward(prediction []float64, labels []float64) (corrections []float64) {
	var cost, zk float64
	corrections = make([]float64, l.prevLayerSize)

	for i, ak := range prediction {
		// Delta rule
		cost = l.costDerivative(ak, labels[i]) * l.actDerivative(l.input[i])
		corrections = append(corrections, cost)
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
