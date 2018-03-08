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
	forward([][]float64) ([][]float64, error)
	backward([]float64) ([]float64, error)
	applyCorrections(float64)
}

type outputLayer interface {
	activation
	cost
	forwardMeasure([][]float64, []float64) ([]float64, float64, error)
	forward(rowInput [][]float64) ([]float64, error)
	backward(prediction, labels []float64) ([]float64, error)
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

	// Exclude bias synapse
	output = make([][]float64, l.nextLayerSize-1)
	for i := 0; i < l.nextLayerSize-1; i++ {
		for j, v := range input {
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
	l.corrections = nil
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
	lastHidden                                  bool
}

func (l *hiddenDense) forward(input [][]float64) (output [][]float64, err error) {
	var inputSum, actValue float64
	output = make([][]float64, l.nextLayerSize)

	// Activated output used at backward propagation, but obviously filled
	// not only after backprop. For correct accumulation of activated
	// output values required cleanup before forward propagation but not after backward.
	l.activated = nil
	l.input = nil
	for _, i := range input {

		// TODO: could be optimized. Don't collect input out of learning process.
		inputSum = 0
		for _, j := range i {
			inputSum += j
		}

		l.input = append(l.input, inputSum)
		actValue, err = l.activate(inputSum)
		if err != nil {
			return
		}

		l.activated = append(l.activated, actValue)
	}

	// TODO: take into account biases
	// There are two cases:
	// A hidden layer is not the last one and a next layer has bias too - we need to return
	// nextLayerSize-1 output slices.
	// A hidden layer is the last one - return nextLayerSize output slices.
	// Also don't multiply activated values by a bias, append bias itself into output slices. 
	var nextLayerBias int
	if !l.lastHidden {
		nextLayerBias = 1
	}
	for i := 0; i < l.nextLayerSize - nextLayerBias; i++ {
		for j, a := range l.activated {
			output[i] = append(output[i], l.synapses[j][i]*a)
		}
		output[i] = append(output[i], l.synapses[l.currLayerSize-1][i])
	}

	return
}

func (l *hiddenDense) backward(eRRors []float64) (nextLayerErrors []float64, err error) {
	// Collect corrections for further forward error propagation
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i, eRR := range eRRors {
		for j, a := range l.activated {
			if l.corrections[j] == nil {
				l.corrections[j] = make([]float64, l.nextLayerSize)
			}
			l.corrections[j][i] += eRR * a
		}
	}

	// Propagate backward
	var eRRSum, actDer float64
	for i, v := range l.input {

		actDer, err = l.actDerivative(v)
		if err != nil {
			return
		}

		eRRSum = 0
		for j, eRR := range eRRors {
			eRRSum += l.synapses[i][j] * eRR
		}
		nextLayerErrors = append(nextLayerErrors, actDer*eRRSum)
	}
	// nextLayerErrors = append(nextLayerErrors, eRRsum)

	return
}

func (l *hiddenDense) applyCorrections(batchSize float64) {
	for i, corr := range l.corrections {
		for j, c := range corr {
			l.synapses[i][j] += l.learningRate * c / batchSize
		}
	}
	l.corrections = nil
}

func newHiddenDense(prev, curr, next int, bias, learningRate float64, activation activation, lastHidden bool) hiddenLayer {
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
		learningRate:  learningRate,
		lastHidden:    lastHidden,
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

func (l *outputDense) forward(rowInput [][]float64) (output []float64, err error) {
	var iSum, actVal float64

	l.input = nil
	for _, raw := range rowInput {
		iSum = 0
		for _, item := range raw {
			iSum += item
		}

		l.input = append(l.input, iSum)
		actVal, err = l.activate(iSum)
		if err != nil {
			return
		}
		output = append(output, actVal)
	}
	return
}

func (l *outputDense) forwardMeasure(rowInput [][]float64, labels []float64) (prediction []float64, cost float64, err error) {
	prediction, err = l.forward(rowInput)
	if err != nil {
		return
	}
	cost = l.countCost(prediction, labels)
	return
}

func (l *outputDense) backward(prediction []float64, labels []float64) (eRRors []float64, err error) {
	var eRR, actDer float64

	for i, pred := range prediction {
		// Delta rule
		actDer, err = l.actDerivative(l.input[i])
		if err != nil {
			return
		}
		eRR = l.costDerivative(pred, labels[i]) * actDer
		eRRors = append(eRRors, eRR)
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
