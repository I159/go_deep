package go_deep

type inputLayer interface {
	forward(setItem []float64) (output float64)
}

type firstHiddenLayer interface {
	activation
	cost
	forward(float64) [][]float64
	backward([][]float64)
	init()
	applyCorrection(float64)
}

type hiddenLayer interface {
	activation
	cost
	forward(arg) return_val
	backward(arg) return_val
	init(arg) return_val
	applyCorrection()
}

type outputLayer interface {
	activation
	cost
	forward(rowInput [][]float64) []float64
	backward(labels []float64) [][]float64
}

type inputDense struct {}

// Optimize input layer and create firstHidden layer and extraHidden layer with different input vector shape
func (l *inputDense) farward(setItem []float64) (output float64) {
	/*
	The Input nodes provide information from the outside world to the 
	network and are together referred to as the “Input Layer”. No computation
	is performed in any of the Input nodes – they just pass on the information to the hidden nodes.
	*/
	for _, i := range set {
		output += i
	}
	output *= .00001
	return
}

type hiddenDenseFirst struct {
	synapseInitializer
	corrections, synapses [][]float64
}

func (l *hiddenDenseFirst) init() {
	l.synapses = l.synapseInitializer.init()
}

func (l *hiddenDenseFirst) forward(input float64) (output [][]float64) {
	// Transition between layers is a matrix reshape. Way or another reshape matrix is required on step of multiplication or sum.
	var j int
	for i := range n.synapses[0] {
		for j = 0; j < currLayerSize - 1; j++ {
			if output[i] == nil {
				output[i] = make([]float64, currLayerSize)
			}
			output[i][j] = l.synapses[j][i] * input
		}
		output[i][j+1] += l.synapses[j+1][i] // Add i bias to the sum of weighted output. Bias doesn't use signal, bias is a weight without input.
	}
	return output
}

func (l *hiddenDenseFirst) backward(corrections [][]float64) {
	for i, corr := range corrections {
		for j, c := range corr {
			l.corrections[i][j] += c
		}
	}
}

func (l *hiddenDenseFirst) applyCorrections(batchSize float64) {
	for i, corr := range l.corrections {
		for j, c := range corr {
			l.synapses[i][j] += l.learningRate * c / batchSize
		}
	}
}


type outputDense struct {
	prevLayerSize int
	input [][]float64
}

func (l *outputDense) forward(rowInput [][]float64) (output []float64) {
	l.out = rowInput
	var iSum float64

	for _, raw := range rowInput {
		iSum = 0
		for _, item := range raw {
			iSum += item
		}
		output = append(output, l.activate(iSum))
	}
	return
}

func (l *outputDense) backward(prediction [][]float64, labels []float64) (corrections [][]float64) {
	var cost, zk float64

	for i, ak := range prediction {
		zk = 0
		for _, aj := range l.input[i] {
			zk += aj // Sum current layer input
		}
		// Delta rule
		cost = n.cost.costDerivative(ak, labels[i]) * n.activation.actDerivative(zk)
		for k := 0; k < l.prevLayerSize; k++ {
			// Corrections vector of the same shape as synapses vector
			correction[k][i] = cost * prediction[i][k]
		}
		// Add bias correction
		correction[l.prevLayerSize][i] = cost
	}
	return
}
