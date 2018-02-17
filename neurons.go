package go_deep

type inputLayer interface {
	forward(arg) return_val
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

func (l *inputDense) farward() {
	
}

type hiddenDense struct {
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
