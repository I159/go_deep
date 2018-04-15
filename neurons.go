package goDeep

import (
	"github.com/I159/go_vectorize"
)

type inputLayer interface {
	synapseInitializer
	forward([]float64) ([]float64, error)
	backward([]float64) error
	applyCorrections(float64) error
}

type hiddenLayer interface {
	activation
	synapseInitializer
	forward([]float64) ([]float64, error)
	backward([]float64) ([]float64, error)
	applyCorrections(float64) error
}

type outputLayer interface {
	activation
	cost
	forwardMeasure([]float64, []float64) ([]float64, float64, error)
	forward(rowInput [][]float64) ([]float64, error)
	backward(prediction, labels []float64) ([]float64, error)
}

type inputDense struct {
	synapseInitializer
	corrections, synapses        [][]float64
	nextLayerSize, currLayerSize int
	learningRate                 float64
	biases, input                []float64
}

func (l *inputDense) forward(input []float64) (output []float64, err error) {
	l.input = input
	output, err = goVectorize.Dot1D2D(input, l.synapses)
	if err != nil {
		return
	}
	if l.biases != nil {
		output, err = goVectorize.Add(output, l.biases)
	}
	return
}

func (l *inputDense) backward(eRRors []float64) (err error) {
	corrections := goVectorize.OuterProduct(eRRors, l.input)
	l.corrections, err = goVectorize.EntrywiseSum(l.corrections, corrections)
	if err != nil {
		return
	}

	if l.biases != nil {
		l.biases, err = goVectorize.Add(l.biases, eRRors)
	}
	return
}

func (l *inputDense) applyCorrections(batchSize float64) (err error) {
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
			return
		}
		for j := 0; j < nextLayerSize; j++ {
			l.synapses[i][j] -= l.learningRate * l.corrections[i][j] / batchSize
		}
	}
	l.corrections = nil

	return
}

func newInputDense(curr, next int, learningRate, bias float64, nextBias bool) inputLayer {
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
	if bias != 0 {
		for i := 0; i < next; i++ {
			layer.biases = append(layer.biases, bias)
		}
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
	activated, input, biases                    []float64
}

func (l *hiddenDense) forward(input []float64) (output []float64, err error) {
	l.input = input
	l.activated, err = goVectorize.ApplyFunction(l.activate, input)
	if err != nil {
		return
	}

	output, err = goVectorize.Dot1D2D(input, l.synapses)
	if err != nil {
		return
	}

	if l.biases != nil {
		output, err = goVectorize.Add(output, l.biases)
	}
	return
}

func (l *hiddenDense) updateCorrections(eRRors []float64) ([][]float64, error) {
	var err error
	corrections := goVectorize.OuterProduct(eRRors, l.input)
	l.corrections, err = goVectorize.EntrywiseSum(l.corrections, corrections)
	if err != nil {
		return nil, err
	}

	if l.biases != nil {
		l.biases, err = goVectorize.Add(l.biases, eRRors)
	}
	return l.corrections, err
}

// Propagate backward from hidden to a previous hidden or input layer
// Single error signal per neuron.
func (l *hiddenDense) backward(eRRors []float64) (prevLayerErrors []float64, err error) {
	l.corrections, err = l.updateCorrections(eRRors)
	if err != nil {
		return
	}

	transposed, err := goVectorize.Transpose(l.synapses, l.nextLayerSize)
	if err != nil {
		return
	}

	errSums, err := goVectorize.Dot1D2D(eRRors, transposed)
	if err != nil {
		return
	}

	actDerivatives, err := goVectorize.ApplyFunction(l.actDerivative, l.input)
	if err != nil {
		return
	}

	prevLayerErrors, err = goVectorize.MultiplyArrays(actDerivatives, errSums)
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
			l.synapses[i][j] -= l.learningRate * l.corrections[i][j] / batchSize
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
		prevLayerSize: prev,
		currLayerSize: curr,
		nextLayerSize: next,
		learningRate:  learningRate,
		nextBias:      nextBias,
		bias:          bias != 0,
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

func (l *outputDense) forward(input []float64) ([]float64, error) {
	l.input = input
	return goVectorize.ApplyFunction(l.activate, input)
}

func (l *outputDense) forwardMeasure(rowInput []float64, labels []float64) (prediction []float64, cost float64, err error) {
	prediction, err = l.forward(rowInput)
	if err != nil {
		return
	}
	cost = l.countCost(prediction, labels)
	return
}

func (l *outputDense) backward(prediction []float64, labels []float64) (eRRors []float64, err error) {
	actDerivatives, err := goVectorize.ApplyFunction(l.actDerivative, l.input)
	if err != nil {
		return
	}

	eRRors, err = goVectorize.Apply2DFunction(prediction, labels)
	if err != nil {
		return
	}

	return goVectorize.MultiplyArrays(eRRors, actDerivatives)
}

func newOutput(prev, curr int, activation activation, cost cost) outputLayer {
	return &outputDense{
		activation:    activation,
		cost:          cost,
		prevLayerSize: prev,
		currLayerSize: curr,
	}
}
