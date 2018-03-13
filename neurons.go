/*
We simply need to calculate the backpropagated error signal that reaches that layer \delta_l
and weight it by the feed-forward signal a_{l-1}feeding into that layer!
*/
package go_deep

type inputLayer interface {
	synapseInitializer
	forward([]float64) ([][]float64, error)
	backward([]float64) error
	applyCorrections(float64) error
}

type hiddenLayer interface {
	activation
	synapseInitializer
	forward([][]float64) ([][]float64, error)
	backward([]float64) ([]float64, error)
	applyCorrections(float64) error
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
	bias                         float64
}

func (l *inputDense) forward(input []float64) (output [][]float64, err error) {
	if err = areSizesConsistent(len(input), l.currLayerSize, len(l.synapses), false); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	l.input = input
	// Exclude bias synapse
	output = make([][]float64, l.nextLayerSize-1)
	for i := 0; i < l.nextLayerSize-1; i++ {
		for j := 0; j < l.currLayerSize; j++ {
			output[i] = append(output[i], l.synapses[j][i]*input[j])
		}
	}
	return
}

func (l *inputDense) backward(eRRors []float64) (err error) {
	if err = checkInputSize(len(eRRors), l.nextLayerSize-1); err == nil {
		//                        				   Bias ^^
		err = checkInputSize(len(l.input), l.currLayerSize)
	}
	if err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i := 0; i < l.nextLayerSize-1; i++ {
		for j := 0; j < l.currLayerSize; j++ {
			if l.corrections[j] == nil {
				l.corrections[j] = make([]float64, l.nextLayerSize)
			}
			l.corrections[j][i] += eRRors[i] * l.input[j]
		}
		if l.bias > 0 {
			if l.corrections[l.currLayerSize-1] == nil {
				l.corrections[l.currLayerSize-1] = make([]float64, l.nextLayerSize)
			}
			l.corrections[l.currLayerSize-1][i] += eRRors[i]
		}
	}
	return
}

func (l *inputDense) applyCorrections(batchSize float64) (err error) {
	if err = areCorrsConsistent(len(l.corrections), l.currLayerSize, len(l.synapses)); err != nil {
		return
	}

	for i := 0; i < l.currLayerSize; i++ {
		if err = areCorrsConsistent(len(l.corrections[i]), l.nextLayerSize, len(l.synapses[i])); err != nil {
			return
		}
		for j := 0; j < l.nextLayerSize; j++ {
			l.synapses[i][j] += l.learningRate * l.corrections[i][j] / batchSize
		}
	}
	l.corrections = nil

	return
}

func newInputDense(curr, next int, learningRate, bias float64) inputLayer {
	layer := &inputDense{
		synapseInitializer: &denseSynapses{
			prev: 1,
			curr: curr,
			next: next,
		},
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  learningRate,
		bias:          bias,
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
	// Input lesser than a layer size because bias has no input.
	if err = areSizesConsistent(len(input), l.currLayerSize, len(l.synapses), true); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	var inputSum, actValue float64
	output = make([][]float64, l.nextLayerSize)

	// Activated output used at backward propagation, but obviously filled
	// not only after backprop. For correct accumulation of activated
	// output values required cleanup before forward propagation but not after backward.
	l.activated = nil
	l.input = nil
	for i := 0; i < l.currLayerSize-1; i++ {
		//for _, i := range input {

		// TODO: could be optimized. Don't collect input out of learning process.
		inputSum = 0
		for _, j := range input[i] {
			inputSum += j
		}

		l.input = append(l.input, inputSum)
		actValue, err = l.activate(inputSum)
		if err != nil {
			return
		}

		l.activated = append(l.activated, actValue)
	}

	var nextLayerBias int
	if !l.lastHidden {
		nextLayerBias = 1
	}
	for i := 0; i < l.nextLayerSize-nextLayerBias; i++ {
		for j := 0; j < l.currLayerSize-1; j++ {
			output[i] = append(output[i], l.synapses[j][i]*l.activated[j])
		}
		output[i] = append(output[i], l.synapses[l.currLayerSize-1][i])
	}
	return
}

func (l *hiddenDense) updateCorrections(eRRors []float64) [][]float64 {
	// Collect corrections for further forward error propagation
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i := 0; i < l.nextLayerSize; i++ {
		for j := 0; j < l.currLayerSize-1; j++ {
			if l.corrections[j] == nil {
				l.corrections[j] = make([]float64, l.nextLayerSize)
			}
			l.corrections[j][i] +=  l.activated[j] * eRRors[i]
		}
		// Apply bias error signal
		if l.corrections[l.currLayerSize-1] == nil {
			l.corrections[l.currLayerSize-1] = make([]float64, l.nextLayerSize)
		}
		l.corrections[l.currLayerSize-1][i] += eRRors[i]
	}
	return l.corrections
}

func (l *hiddenDense) backward(eRRors []float64) (prevLayerErrors []float64, err error) {
	// Propagate backward from hidden to a previous hidden or input layer
	// Single error signal per neuron.
	l.corrections = l.updateCorrections(eRRors)

	// Bias is not connected with previous layer so exclude the last synapse
	// Which is a bias for error signal computation.
	var eRRSum, actDer float64
	for i := 0; i < l.currLayerSize-1; i++ {

		actDer, err = l.actDerivative(l.input[i])
		if err != nil {
			return
		}

		eRRSum = 0
		for j := 0; j < l.nextLayerSize; j++ {
			eRRSum += l.synapses[i][j] * eRRors[j]
		}
		prevLayerErrors = append(prevLayerErrors, actDer*eRRSum)
	}
	return
}

// FIXME: refactor this crap. Should be implemented in a single place.
func (l *hiddenDense) applyCorrections(batchSize float64) (err error) {
	if err = areCorrsConsistent(len(l.corrections), l.currLayerSize, len(l.synapses)); err != nil {
		return
	}

	for i := 0; i < l.currLayerSize; i++ {
		if err = areCorrsConsistent(len(l.corrections[i]), l.nextLayerSize, len(l.synapses[i])); err != nil {
			return
		}
		for j := 0; j < l.nextLayerSize; j++ {
			l.synapses[i][j] += l.learningRate * l.corrections[i][j] / batchSize
		}
	}
	l.corrections = nil

	return
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
	prevLayerSize, currLayerSize int
}

func (l *outputDense) forward(rowInput [][]float64) (output []float64, err error) {
	if err = checkInputSize(len(rowInput), l.currLayerSize); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

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
		currLayerSize: curr,
	}
}
