package main

type synapsesOps interface {
	intializeSynapses(hidden, input, output int) [][]float64
	addBiases(synapses [][]float64) [][]float64
	// TODO: correct synapses using batches
}
