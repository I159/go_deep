package go_deep

import "fmt"

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
	return nil
}

func (l *Perceptron) applyCorrections(batchSize float64) {
	for _, l := range l.hidden {
		l.applyCorrections(batchSize)
	}
	l.input.applyCorrections(batchSize)
}

func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) ([]float64, error) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	var localCost float64
	var costGradient []float64

	for j := 0; j <= epochs; j++ {
		fmt.Printf("Epochs: %d\n", j)
		for i, v := range set {
			if batchCounter >= batchSize {
				n.applyCorrections(float64(batchSize))
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
	}
	return costGradient, nil
}

func (n *Perceptron) forward(rowInput []float64) ([]float64, error) {
	var fwdProp [][]float64
	var err error

	fwdProp = n.input.forward(rowInput)
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
	fwdProp = n.input.forward(rowInput)
	for _, l := range n.hidden {
		fwdProp, err = l.forward(fwdProp)
		if err != nil {
			return nil, 0, err
		}
	}
	return n.output.forwardMeasure(fwdProp, labels)
}

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

type InputShape struct {
	Size         int
	LearningRate float64
}
type HiddenShape struct {
	Size               int
	LearningRate, Bias float64
	Activation         activation
}

type OutputShape struct {
	Size       int
	Activation activation
	Cost       cost
}

func NewPerceptron(inputShape InputShape, hiddenShapes []HiddenShape, outputShape OutputShape) network {
	return &Perceptron{
		input: newInputDense(inputShape.Size, hiddenShapes[0].Size, inputShape.LearningRate),
		hidden: []hiddenLayer{
			newHiddenDense(
				inputShape.Size,
				hiddenShapes[0].Size,
				outputShape.Size,
				hiddenShapes[0].Bias,
				hiddenShapes[0].LearningRate,
				hiddenShapes[0].Activation,
			),
		},
		output: newOutput(hiddenShapes[0].Size, outputShape.Size, outputShape.Activation, outputShape.Cost),
	}
}
