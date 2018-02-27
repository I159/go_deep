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
	activation
	synapseInitializer
	forward([][]float64) [][]float64
	backward([]float64) []float64
	applyCorrections(float64)
}

type outputLayer interface {
	activation
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

	output = make([][]float64, l.nextLayerSize)
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

	for i, eRR := range eRRors {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}
		for j, a := range l.input {
			l.corrections[j][i] += eRR * a
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

func newInputDense(curr, next int, learningRate float64) inputLayer {
	layer := &inputDense{
		synapseInitializer: &denseSynapses{
			prev: 1,
			curr: curr,
			next: next,
		},
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  learningRate,
	}
	layer.synapses = layer.init()
	return layer
}

type hiddenDense struct {
	activation
	synapseInitializer
	prevLayerSize, currLayerSize, nextLayerSize int
	learningRate                                float64
	corrections, synapses                       [][]float64
	activated, input                            []float64
}

func (l *hiddenDense) forward(input [][]float64) (output [][]float64) {
	var inputSum float64
	output = make([][]float64, l.nextLayerSize)

	for _, i := range input {
		inputSum = 0
		for _, j := range i {
			inputSum += j
		}
		l.input = append(l.input, inputSum)
		l.activated = append(l.activated, l.activate(inputSum))
	}

	for i := 0; i < l.nextLayerSize; i++ {
		for j, a := range l.activated {
			if output[i] == nil {
				output[i] = make([]float64, l.currLayerSize)
			}
			// Transition between layers is a matrix reshape. Way or another reshape matrix is required on step of multiplication or sum.
			output[i][j] = l.synapses[j][i] * a
		}
		output[i][l.currLayerSize-1] = l.synapses[l.currLayerSize-1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
	}

	return output
}

func (l *hiddenDense) backward(eRRors []float64) (nextLayerErrors []float64) {
	// Collect corrections for further forward error propagation
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i, eRR := range eRRors {
		if l.corrections[i] == nil {
			l.corrections[i] = make([]float64, l.nextLayerSize)
		}
		for j, a := range l.activated {
			l.corrections[j][i] = eRR * a
		}
	}

	// Propagate backward
	var eRRSum float64
	for i := range l.synapses {
		actDer := l.actDerivative(l.input[i])
		eRRSum = 0
		for j, eRR := range eRRors {
			eRRSum += l.synapses[i][j] * eRR
		}
		nextLayerErrors = append(nextLayerErrors, actDer*eRRSum)
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

func newHiddenDense(prev, curr, next int, bias, learningRate float64, activation activation) hiddenLayer {
	layer := &hiddenDense{
		activation: activation,
		synapseInitializer: &hiddenDenseSynapses{
			denseSynapses{
				prev: prev,
				curr: curr,
				next: next,
			},
			bias,
		},
		prevLayerSize: prev,
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  -learningRate,
	}
	layer.synapses = layer.init()
	return layer
}

type outputDense struct {
	activation
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

func (l *outputDense) backward(prediction []float64, labels []float64) (eRRors []float64) {
	var cost float64
	eRRors = make([]float64, l.prevLayerSize)

	for i, pred := range prediction {
		// Delta rule
		cost = l.costDerivative(pred, labels[i]) * l.actDerivative(l.input[i])
		eRRors = append(eRRors, cost)
	}
	return
}

func newOutput(prev, curr int, activation activation, cost cost) outputLayer {
	return &outputDense{
		activation:    activation,
		cost:          cost,
		prevLayerSize: prev,
	}
}
