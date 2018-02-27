package go_deep

type Perceptron struct {
	input  inputLayer
	hidden []hiddenLayer
	output outputLayer
}

func (n *Perceptron) backward(prediction []float64, labels []float64) {
	var backpropErrs []float64
	backpropErrs = n.output.backward(prediction, labels)
	for _, l := range n.hidden {
		backpropErrs = l.backward(backpropErrs)
	}
}

func (l *Perceptron) applyCorrections(batchSize float64) {
	for _, l := range l.hidden {
		l.applyCorrections(batchSize)
	}
	l.input.applyCorrections(batchSize)
}

func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) (costGradient []float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	var localCost float64

	for j := 0; j <= epochs; j++ {
		for i, v := range set {
			if batchCounter >= batchSize {
				n.applyCorrections(float64(batchSize))
				costGradient = append(costGradient, localCost/float64(batchSize))
				batchCounter = 0
				localCost = 0
			}
			prediction, cost := n.forwardMeasure(v, labels[i])
			localCost += cost
			n.backward(prediction, labels[i])
			batchCounter++
		}
	}
	return
}

func (n *Perceptron) forward(rowInput []float64) []float64 {
	// NOTE: this is a single layer implementation
	var fwdProp [][]float64
	fwdProp = n.input.forward(rowInput)
	for _, l := range n.hidden {
		fwdProp = l.forward(fwdProp)
	}
	return n.output.forward(fwdProp)
}

func (n *Perceptron) forwardMeasure(rowInput, labels []float64) (prediction []float64, cost float64) {
	var fwdProp [][]float64
	fwdProp = n.input.forward(rowInput)
	for _, l := range n.hidden {
		fwdProp = l.forward(fwdProp)
	}
	return n.output.forwardMeasure(fwdProp, labels)
}

func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64) {
	var pred []float64

	for _, v := range set {
		pred = n.forward(v)
		prediction = append(prediction, pred)
	}
	return
}


type HiddenShape struct {
	Size int
	LearningRate, Bias float64
	Activation Activation
}

type OutputShape struct {
	Size int
	LearningRate float64
	Activation Activation
	Cost cost
}

func NewPerceptron(inputSize int, hiddenShapes []HiddenShape, outputShape OutputShape) network {
	return &Perceptron{
		input: &inputDense{},
		hidden: []hiddenLayer{
			newHiddenDense(
				inputSize,
				hiddenShapes[0].Size,
				OutputShape.Size,
				hiddenShapes[0].LearningRate,
				hiddenShapes[0].Activation,
			),
		},
		output: newOutput(hiddenShapes[0].Size, outputShape.Size, outputShape.Activation, outputShape.Cost),
	}
}
