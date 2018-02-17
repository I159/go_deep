package go_deep

type inputLayer interface {
	forward(setItem []float64) (output float64)
}

type firstHiddenLayer interface {
	forward(float64) [][]float64
	backward()
	init()
}

type hiddenLayer interface {
	forward(arg) return_val
	backward(arg) return_val
	init(arg) return_val
}

type outputLayer interface {
	forward(arg) return_val
	backward(arg) return_val
}

type inputDense struct {}

// Optimize input layer and create firstHidden layer and extraHidden layer with different input vector shape
func (l *inputDense) farward(setItem []float64) (output float64) {
	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		output += i
	}
	output *= .00001
	return
}

type hiddenDenseFirst struct {
	synapseInitializer
	synapses [][]float64
	prevOut [][]float64
}

func (l *hiddenDense) init() error {
	
}

func (l *hiddenDense) forward() error {
	
}

func (l *hiddenDense) backward() error {
	
}

type outputDense struct {
	prevOut [][]float64
	actOut	[][]float64
}

func (l *outputDense) forward() error {
	
}

func (n *outputDense) backward() error {
	
}
