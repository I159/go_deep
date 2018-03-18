/*
Neurons layers recognition and learning.
*/
package go_deep

import "fmt"

type layerShaper1D interface {
	checkInput(input []float64) error
	isBias() bool
}

type layerShaper2D interface {
	checkInput(input [][]float64) error
	isBias() bool
}

type inputLayer interface {
	synapseInitializer
	layerShaper1D
	forward([]float64) ([][]float64, error)
	backward([]float64) error
	applyCorrections(float64)
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

type shapeInput struct {
	bias, nextBias               bool
	nextLayerSize, currLayerSize int
}

func (c *shapeInput) isBias() bool {
	return c.bias
}

func (l *shapeInput) checkInput(input []float64) (err error) {
	var currLayerSize = l.currLayerSize
	if l.bias {
		currLayerSize--
	}
	if len(input) != currLayerSize {
		err = locatedError{
			fmt.Sprintf(
				"Input is not relevant. Input: %d Layer: %d Bias: %t",
				len(input),
				l.currLayerSize,
				l.bias,
			),
		}
		lockErr := err.(locatedError)
		err = lockErr.freeze()
	}
	return
}

type shapeHidden struct {
	bias, nextBias                              bool
	prevLayerSize, nextLayerSize, currLayerSize int
}

func (c *shapeHidden) isBias() bool {
	return c.bias
}

func (l *shapeHidden) checkInput(input [][]float64) (err error) {
	var currLayerSize = l.currLayerSize
	if l.bias {
		currLayerSize--
	}
	if len(input) != currLayerSize {
		err = locatedError{
			fmt.Sprintf(
				"Input is not relevant. Input: %d Layer: %d Bias: %t",
				len(input),
				l.currLayerSize,
				l.bias,
			),
		}
		lockErr := err.(locatedError)
		err = lockErr.freeze()
	}
	return
}

type inputDense struct {
	synapseInitializer
	layerShaper1D
	corrections, synapses [][]float64
	learningRate          float64
	input                 []float64
}

func (l *inputDense) forward(input []float64) (output [][]float64, err error) {
	/*
		Propagate input signal forward

		Receive input signal and check it to be consistent with current layer size.
		Keep input for further backward propagation corrections computation. Multiply
		input signal to synapses vector. Synapses is a 2D vector. A second dimension
		vector is synapses of an input layer neuron with all neurons of a hidden
		layer. If bias exists on an input layer then bias synapses placed in the last
		synapses second dimension layer will be augmented to output signal and passed
		to the first hidden layer.
	*/
	if err = l.checkInput(input); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	l.input = input

	output = transMul1dTo2d(input, l.synapses)
	if l.isBias() {
		output = augment(output, l.synapses[len(l.synapses)-1])
	}
	return
}

func (l *inputDense) backward(eRRors []float64) (err error) {
	/*
		The last step of backward propagation

		Receive errors from a previous layer. Multiply error signal to saved input.
		If bias exists append obtained corrections by error signal. If corrections
		exist update it otherwise assign corrections by newly obtained corrections
		vector.
	*/
	if err = l.checkInput(eRRors); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	corrections := dotProduct1d(l.input, eRRors)
	if l.isBias() {
		corrections = append(corrections, eRRors)
	}

	if l.corrections == nil {
		l.corrections = corrections
	} else {
		l.corrections = add2D(l.corrections, corrections)
	}
	return
}

// TODO: apply decomposition
func (l *inputDense) applyCorrections(batchSize float64) {
	for i, v := range l.synapses {
		for j := range v {
			l.synapses[i][j] += l.learningRate * l.corrections[i][j] / batchSize
		}
	}
	l.corrections = nil
}

func newInputDense(curr, next int, learningRate, bias float64, nextBias bool) inputLayer {
	layer := &inputDense{
		synapseInitializer: &denseSynapses{
			prev:     1,
			curr:     curr,
			next:     next,
			bias:     bias,
			nextBias: nextBias,
		},
		layerShaper1D: &shapeInput{
			currLayerSize: curr,
			nextLayerSize: next,
			bias:          bias != 0,
			nextBias:      nextBias,
		},
		learningRate: learningRate,
	}
	layer.synapses = layer.init()
	return layer
}

type hiddenDense struct {
	activation
	synapseInitializer
	layerShaper2D
	learningRate          float64
	corrections, synapses [][]float64
	activated, input      []float64
}

func (l *hiddenDense) forward(input [][]float64) (output [][]float64, err error) {
	if err = l.checkInput(input); err != nil {
		lockErr:=err.(locatedError)
		err = lockErr.freeze()
		return
	}

	l.input = transSum2dTo1d(input)
	// TODO: separate into applyOperaion vector operation
	for i, v := range l.input {
		l.activated[i] , err = l.activate(v)
		if err != nil {
			return
		}
	}

	// TODO: separate into a vector operation
	for i := 0; i < nextLayerSize; i++ {
		for j := 0; j < currLayerSize; j++ {
			output[i] = append(output[i], l.synapses[j][i]*l.activated[j])
		}
		if l.bias {
			output[i] = append(output[i], l.synapses[currLayerSize][i])
		}
	}
	return
}

func (l *hiddenDense) updateCorrections(eRRors []float64) [][]float64 {
	currLayerSize := l.currLayerSize
	if l.bias {
		currLayerSize--
	}
	nextLayerSize := l.nextLayerSize
	if l.nextBias {
		nextLayerSize--
	}

	// Collect corrections for further forward error propagation
	if l.corrections == nil {
		l.corrections = make([][]float64, l.currLayerSize)
	}

	for i := 0; i < nextLayerSize; i++ {
		for j := 0; j < currLayerSize; j++ {
			if l.corrections[j] == nil {
				l.corrections[j] = make([]float64, l.nextLayerSize)
			}
			l.corrections[j][i] += l.activated[j] * eRRors[i]
		}
		// Apply bias error signal
		if l.corrections[currLayerSize] == nil {
			l.corrections[currLayerSize] = make([]float64, nextLayerSize)
		}
		l.corrections[currLayerSize][i] += eRRors[i]
	}
	return l.corrections
}

func (l *hiddenDense) backward(eRRors []float64) (prevLayerErrors []float64, err error) {
	// Propagate backward from hidden to a previous hidden or input layer
	// Single error signal per neuron.
	l.corrections = l.updateCorrections(eRRors)

	nextLayerSize := l.nextLayerSize
	if l.nextBias {
		nextLayerSize--
	}
	currLayerSize := l.currLayerSize
	if l.bias {
		currLayerSize--
	}

	// Bias is not connected with previous layer so exclude the last synapse
	// From both previous layer errors vector and a current one.
	var eRRSum, actDer float64
	for i := 0; i < currLayerSize; i++ {

		actDer, err = l.actDerivative(l.input[i])
		if err != nil {
			return
		}

		eRRSum = 0
		for j := 0; j < nextLayerSize; j++ {
			eRRSum += l.synapses[i][j] * eRRors[j]
		}
		prevLayerErrors = append(prevLayerErrors, actDer*eRRSum)
	}
	return
}

// FIXME: refactor this crap. Should be implemented in a single place.
func (l *hiddenDense) applyCorrections(batchSize float64) (err error) {
	if err = areCorrsConsistent(len(l.corrections), l.currLayerSize, len(l.synapses)); err != nil {
		lockErr := err.(locatedError)
		err = lockErr.freeze()
		return
	}

	nextLayerSize := l.nextLayerSize
	if l.nextBias {
		nextLayerSize--
	}

	for i := 0; i < l.currLayerSize; i++ {
		if err = areCorrsConsistent(len(l.corrections[i]), nextLayerSize, len(l.synapses[i])); err != nil {
			lockErr := err.(locatedError)
			err = lockErr.freeze()
			return
		}
		for j := 0; j < nextLayerSize; j++ {
			l.synapses[i][j] += l.learningRate * l.corrections[i][j] / batchSize
		}
	}
	l.corrections = nil

	return
}

func newHiddenDense(prev, curr, next int, bias, learningRate float64, activation activation, nextBias bool) hiddenLayer {
	layer := &hiddenDense{
		activation: activation,
		synapseInitializer: &hiddenDenseSynapses{
			denseSynapses{
				prev:     prev,
				curr:     curr,
				next:     next,
				bias:     bias,
				nextBias: nextBias,
			},
		},
		layerShaper2D: &shapeHidden{
			prevLayerSize: prev,
			currLayerSize: curr,
			nextLayerSize: next,
			nextBias:      nextBias,
			bias:          bias != 0,
		},
		learningRate: learningRate,
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
