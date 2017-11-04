package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const SCALING_BASE = 0.7
const INPUT = 784
const OUTPUT = 10
const HIDDEN = 64
const INPUT_BAIAS = .5

// Create random 2D vector. Implement pre-init synapses
func random() (out [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < HIDDEN; i++ {
		out = append(out, []float64{})
		for j := 0; j < OUTPUT; j++ {
			out[i] = append(out[i], rand.Float64()-0.5) // Float64 returns, as a float64, a pseudo-random number in [0.0,1.0).
		}
	}
	return
}

func NguyenWiderow() [][]float64 {
	beta := SCALING_BASE * math.Pow(HIDDEN, 1.0/INPUT)
	randSynapses := random()

	var norm float64
	for _, i := range randSynapses {
		norm = 0
		for _, j := range i {
			norm += j * j
		}
		norm = math.Sqrt(norm)
		for j, k := range i {
			i[j] = (k * beta) / norm
		}
	}
	return randSynapses
}

func addBiases(synapses [][]float64) [][]float64 {
	synapses = append(synapses, make([]float64, OUTPUT))
	for i := 0; i < OUTPUT; i++ {
		synapses[HIDDEN][i] = 1.
	}

	for _, i := range synapses {
		fmt.Println(i)
	}
	return synapses
}

// For example Sygmoid
func sygmoid(n float64) float64 {
	return 1 / (1 + math.Exp(n))
}

func forward(set []float64, synapses [][]float64) (output []float64) {
	var iSum, oSum float64

	for _, i := range set {
		iSum += i
	}
	iSum = sygmoid(iSum)

	li := len(synapses[0])
	lm := len(synapses)
	for i := 0; i < li; i++ {
		oSum = 0
		for j := 0; j < lm; i++ {
			oSum += synapses[j][i] * iSum
		}
		output = append(output, sygmoid(oSum))
	}
	return
}

func quadratic_cost(a, e []float64) float64 {
	var sum float64
	for i, v := range a {
		sum += v - e[i]
	}
	return sum / 2.
}

func quadratic_derivative(a, e float64) float64 {
	return a - e
}

// n: weighted sum of j-1 layer activations. A.k.a. j layer k neuron input
func sygmoid_derivative(n float64) float64 {
	return sygmoid(n) * (1 - sygmoid(n))
}

// TODO: add biases per layer

func backward(out, labels []float64, inp, synapses [][]float64) [][]float64 {
	var cost, sumInp float64
	for i, v := range out {
		for _, j := range inp[i] {
			sumInp += j
		}
		cost = quadratic_derivative(v, labels[i]) * sygmoid_derivative(sumInp)
		for k := range synapses {
			// Not to keep weights corrections id possible only for stochastic descent
			synapses[i][k] += cost * inp[i][k]
		}
	}
	return synapses
}

func restore(id int) [][]float64 {
	return [][]float64{}
}

func main() {
	synapses := NguyenWiderow()
	synapses = addBiases(synapses)
	fmt.Println(forward(set, synapses))
}
