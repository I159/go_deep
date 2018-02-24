package go_deep

type Perceptron struct {
	input       inputLayer
	hiddenFirst firstHiddenLayer
	//hidden      []hiddenLayer
	output outputLayer
}

func (n *Perceptron) backward(prediction []float64, labels []float64) {
	n.hiddenFirst.backward(
		n.output.backward(prediction, labels),
	)
}

func (n *Perceptron) Learn(set, labels [][]float64, epochs, batchSize int) (costGradient []float64) {
	// Use Recognize loop to get recognition results and hidden layer intermediate results.
	// Loop backward using obtained results for learning
	var batchCounter int
	var localCost float64

	for j := 0; j <= epochs; j++ {
		for i, v := range set {
			if batchCounter >= batchSize {
				n.hiddenFirst.applyCorrections(float64(batchSize))
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
	return n.output.forward(
		n.hiddenFirst.forward(
			n.input.forward(rowInput),
		),
	)
}

func (n *Perceptron) forwardMeasure(rowInput, labels []float64) (prediction []float64, cost float64) {
	res := n.hiddenFirst.forward(
		n.input.forward(rowInput),
	)
	return n.output.forwardMeasure(res, labels)
}

func (n *Perceptron) Recognize(set [][]float64) (prediction [][]float64) {
	var pred []float64

	for _, v := range set {
		pred = n.forward(v)
		prediction = append(prediction, pred)
	}
	return
}

type Shape struct {
	InputSize           int
	HiddenSizes         []int // TODO: use it in multilayer
	OutputSize          int
	HiddenLearningRates []float64
	HiddenActivations   []Activation
	OutputActivation    Activation
	Cost                cost
}

func NewPerceptron(shape Shape) network {
	return &Perceptron{
		input: &inputDense{},
		hiddenFirst: newFirstHidden(
			shape.InputSize,
			shape.HiddenSizes[0],
			shape.OutputSize,
			shape.HiddenLearningRates[0],
			shape.HiddenActivations[0],
		),
		output: newOutput(shape.HiddenSizes[0], shape.OutputSize, shape.OutputActivation, shape.Cost),
	}
}
