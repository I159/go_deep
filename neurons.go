package go_deep

type inputLayer interface {
	forward(setItem []float64) (output float64)
}

type firstHiddenLayer interface {
	activation
	cost
	forward(float64) [][]float64
	backward()
	init()
}

type hiddenLayer interface {
	activation
	cost
	forward(arg) return_val
	backward(arg) return_val
	init(arg) return_val
}

type outputLayer interface {
	activation
	cost
	forward(rowInput [][]float64) ([]float64)
	backward(arg) return_val
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
	synapses [][]float64
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

func (l *hiddenDense) backward() error {
}

type outputDense struct {
	out	[][]float64
}

func (l *outputDense) forward(rowInput [][]float64) (output []float64)
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

func (n *outputDense) backward() error {
	
}
