package main

type network interface {
	synapsesOps
	activation
	cost
	forward()
	backward()
	learn()
	recognize()
}

type Perceptron struct {
	synapses denseSynapses
	activation sygmoid
	cost quadratic
}

func (n *Perceptron) forward() {} 
func (n *Perceptron) backward() {}
func (n *Perceptron) recognize() {}
