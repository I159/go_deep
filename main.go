package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"
)

const TEST_SET_SIZE = 10000

func getTrainigImages() (set [][]float64, err error) {
	imagesFile, err := os.Open("t10k-images-idx3-ubyte")
	if err != nil {
		return
	}

	var offset int
	cursor := 512

	image := make([]byte, INPUT)
	f64image := make([]float64, INPUT)
	lim := TEST_SET_SIZE * INPUT
	for cursor < lim {
		offset, err = imagesFile.ReadAt(image, int64(cursor))
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return
		}
		cursor += offset
		for i, v := range image {
			f64image[i] = float64(v)
		}
		set = append(set, f64image)
	}
	return
}

func getTrainingLabels() (labels []float64, err error) {
	fp, err := os.Open("t10k-labels-idx1-ubyte")
	if err != nil {
		return
	}
	defer fp.Close()
	labelsBuffer := bufio.NewReader(fp)

	magic, err := binary.ReadUvarint(labelsBuffer)
	if err != nil {
		return
	}
	if magic != 2049 {
		err = fmt.Errorf("Wrong magic number %d", magic)
		return
	}
	fp.Seek(32, 1)

	c, err := binary.ReadUvarint(labelsBuffer)
	if err != nil {
		return
	}
	count := int(c)
	fp.Seek(32, 1)

	var labe uint64
	for i := 0; i < count*8; i += 8 {
		labe, err = binary.ReadUvarint(labelsBuffer)
		if err != nil {
			return
		}
		labels = append(labels, float64(labe))
	}
	return
}

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

// TODO: find out how use biases in forward propagation
func addBiases(synapses [][]float64) [][]float64 {
	synapses = append(synapses, make([]float64, OUTPUT))
	for i := 0; i < OUTPUT; i++ {
		synapses[HIDDEN][i] = 1.
	}
	return synapses
}

// For example Sygmoid
func sygmoid(n float64) float64 {
	return 1 / (1 + math.Exp(n))
}

func forward(set []float64, synapses [][]float64) (output []float64) {
	var iSum, oSum float64

	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}
	iSum = sygmoid(iSum) // Activation of signal at a hidden layer

	li := len(synapses[0]) - 1 // Count of synapses between hidden and output layer
	lm := len(synapses) - 2    // Count of neurons of a hidden layer apart from bias neuron
	for i := 0; i < li; i++ {
		oSum = 0
		for j := 0; j < lm; j++ {
			// Output layer neurons sums weighted signal
			// TODO: save hidden layer output
			oSum += synapses[j][i] * iSum // Output signal of a hidden layer
		}
		// Apply a bias
		oSum += synapses[lm+1][i] // Bias doesn't use weights. Bias is a weight without a signal.
		// Output layer applies activation function and returns per neuron single prediction value
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
// (ak - tk)g`(zk)aj
// ak - output of an output layer k
// zk - output layer k input (sum of hidden layer outputs)
// tk - correct answer for an output neuron
// aj - output of a hidden layer neuron
func backward(out, labels []float64, hiddenOut, synapses [][]float64) [][]float64 {
	var cost, zk float64
	for i, ak := range out { // outputs of an out layer
		zk = 0                            // out layer k neuron input (sum of a hidden layer outputs)
		for _, aj := range hiddenOut[i] { // Weighted outputs of a hidden layer k neuron
			// Count k neuron of out layer inout (sum output layer input value)
			zk += aj
		}
		// Count an error derivative
		cost = quadratic_derivative(ak, labels[i]) * sygmoid_derivative(zk)
		for k := range synapses {
			// Multiply an error by output of an appropriate hidden neuron
			// Correct a synapse immediately (Stochastic gradient)
			synapses[k][i] += cost * hiddenOut[k][i]
		}
		// TODO: correct biases
	}
	return synapses
}

func restore(id int) [][]float64 {
	return [][]float64{}
}

func main() {
	synapses := NguyenWiderow()
	synapses = addBiases(synapses)
	set, err := getTrainigImages()
	if err != nil {
		return
	}
	labels, err := getTrainingLabels()
	if err != nil {
		return
	}
	fmt.Println(labels)
	prediction := forward(set[0], synapses)
	fmt.Println(backward(prediction, labels, hiddenOut, synapses))
}
