package goDeep

import "fmt"

/*
Perceptron is MLP implementation of a Network interface.
*/
type Perceptron struct {
	input  inputLayer
	hidden []hiddenLayer
	output outputLayer
}

func (n *Perceptron) backward(prediction []float64, labels []float64) (err error) {
	var backpropErrs []float64

	backpropErrs, err = n.output.backward(prediction, labels)
	if err != nil {
		return err
	}

	for _, l := range n.hidden {
		backpropErrs, err = l.backward(backpropErrs)
		if err != nil {
			return err
		}
	}
	return n.input.backward(backpropErrs)
}

func (n *Perceptron) applyCorrections(batchSize float64) (err error) {
	for _, l := range n.hidden {
		if err = l.applyCorrections(batchSize); err != nil {
			return
		}
	}
	return n.input.applyCorrections(batchSize)
}

// Learn generalization of back propagation for all layers defined in the network
func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) (costGradient []float64, err error) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	var localCost float64

	for j := 0; j < epochs; j++ {
		fmt.Printf("Epochs: %d\n", j+1)
		for i, v := range set {
			if batchCounter >= batchSize {
				if err = n.applyCorrections(float64(batchSize)); err != nil {
					return
				}
				costGradient = append(costGradient, localCost/float64(batchSize))
				batchCounter = 0
				localCost = 0
			}
			prediction, cost, err := n.forwardMeasure(v, labels[i])
			if err != nil {
				return nil, err
			}

			localCost += cost
			err = n.backward(prediction, labels[i])
			if err != nil {
				return nil, err
			}

			batchCounter++
		}

		// Don't forget to apply correction of the last iteration
		// in case of incomplete batch.
		if err = n.applyCorrections(float64(batchCounter)); err != nil {
			return
		}
	}
	return costGradient, nil
}

func (n *Perceptron) forward(rowInput []float64) ([]float64, error) {
	var fwdProp [][]float64
	var err error

	fwdProp, err = n.input.forward(rowInput)
	if err != nil {
		return nil, err
	}

	for _, l := range n.hidden {
		fwdProp, err = l.forward(fwdProp)
		if err != nil {
			return nil, err
		}
	}
	return n.output.forward(fwdProp)
}

func (n *Perceptron) forwardMeasure(rowInput, labels []float64) (prediction []float64, cost float64, err error) {
	var fwdProp [][]float64

	fwdProp, err = n.input.forward(rowInput)
	if err != nil {
		return nil, 0, err
	}

	for _, l := range n.hidden {
		fwdProp, err = l.forward(fwdProp)
		if err != nil {
			return nil, 0, err
		}
	}
	return n.output.forwardMeasure(fwdProp, labels)
}

// Recognize is a generalization of forward propagation for all layers defined in the network
func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64, err error) {
	var pred []float64

	for _, v := range set {
		pred, err = n.forward(v)
		if err != nil {
			return nil, err
		}
		prediction = append(prediction, pred)
	}
	return
}

// InputShape is an intuitive input layer representation. Designed to
//pass declaration arguments in intuitive form.
type InputShape struct {
	Size               int
	LearningRate, Bias float64
}

// HiddenShape is intuitive hidden layer representation. Designed to
// pass declaration arguments in intuitive form.
type HiddenShape struct {
	Size               int
	LearningRate, Bias float64
	Activation         activation
}

// OutputShape is intuitive output layer representation. Designed to
// pass declaration arguments in intuitive form.
type OutputShape struct {
	Size       int
	Activation activation
	Cost       cost
}

// NewPerceptron is a MLP initializer
func NewPerceptron(inputShape InputShape, hiddenShapes []HiddenShape, outputShape OutputShape) Network {
	return &Perceptron{
		input: newInputDense(
			inputShape.Size,
			hiddenShapes[0].Size,
			inputShape.LearningRate,
			inputShape.Bias,
			hiddenShapes[0].Bias != 0,
		),
		hidden: []hiddenLayer{
			newHiddenDense(
				inputShape.Size,
				hiddenShapes[0].Size,
				outputShape.Size,
				hiddenShapes[0].Bias,
				hiddenShapes[0].LearningRate,
				hiddenShapes[0].Activation,
				false,
			),
		},
		output: newOutput(hiddenShapes[0].Size, outputShape.Size, outputShape.Activation, outputShape.Cost),
	}
}
