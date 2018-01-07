package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
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
	defer imagesFile.Close()

	var magic int32
	err = binary.Read(imagesFile, binary.BigEndian, &magic)
	if err != nil {
		return
	}
	if magic != 2051 {
		err = fmt.Errorf("Wrong magic number: %d. Expects: %d.", magic, 2051)
		return
	}

	var rows int32
	err = binary.Read(imagesFile, binary.BigEndian, &rows)
	if err != nil {
		return
	}

	var lines int32
	err = binary.Read(imagesFile, binary.BigEndian, &magic)
	if err != nil {
		return
	}

	inputSize := lines * rows

	image := make([]byte, inputSize)
	f64image := make([]float64, inputSize) // Prepared for recognition
	// FIXME: Read dosn't seek descriptor. The loop is infinit.
	for err == nil {
		_, err = imagesFile.Read(image)
		for i, v := range image {
			f64image[i] = float64(v)
		}
		set = append(set, f64image)
	}
	if err == io.EOF {
		err = nil
	}
	return
}

func getTrainingLabels() (labels [][]float64, err error) {
	// FIXME: somewhere 10001 labels taken instead of 10000
	fp, err := os.Open("t10k-labels-idx1-ubyte")
	if err != nil {
		return
	}
	defer fp.Close()

	var magic int32
	err = binary.Read(fp, binary.BigEndian, &magic)
	if err != nil {
		return
	}

	if magic != 2049 {
		err = fmt.Errorf("Wrong magic number: %d. Expects: %d.", magic, 2049)
		return
	}

	var count int32
	err = binary.Read(fp, binary.BigEndian, &count)
	if err != nil {
		return
	}

	for err == nil {
		var labe uint8
		err = binary.Read(fp, binary.BigEndian, &labe)
		var hotEncoding []float64
		for i := 0; i < OUTPUT; i++ {
			if int(labe) == i {
				hotEncoding = append(hotEncoding, 1)
			} else {
				hotEncoding = append(hotEncoding, 0)
			}
		}
		labels = append(labels, hotEncoding)
	}
	if err == io.EOF {
		fmt.Println("End")
		err = nil
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

func forward(set []float64, synapses [][]float64) (output []float64, hiddenOut [][]float64) {
	var iSum, oSum float64

	// Each neuron of a first hidden layer receives all signals from input layer
	// and sums it. Input layer doesn't change input signal
	for _, i := range set {
		iSum += i * .00001 // Lowering of signal values to prevent overflow
	}

	iSum = sygmoid(iSum) // Activation of signal at a hidden layer
	lm := len(synapses)    // Count of neurons of a hidden layer apart from bias neuron

	for i := range synapses[0]{
		var outLine []float64
		oSum = 0

		for j := range synapses {
			jIOut := synapses[j][i] * iSum
			oSum += jIOut
			outLine = append(outLine, jIOut)
		}

		hiddenOut = append(hiddenOut, outLine)
		// Apply a bias
		oSum += synapses[lm-1][i] // Bias doesn't use weights. Bias is a weight without a signal.
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
	exceptBiases := synapses[:HIDDEN]
	for i, ak := range out { // outputs of an out layer
		zk = 0                            // out layer k neuron input (sum of a hidden layer outputs)
		for _, aj := range hiddenOut[i] { // Weighted outputs of a hidden layer k neuron
			// Count k neuron of out layer input (sum output layer input value)
			zk += aj
		}
		// Count an error derivative using delta rule
		cost = quadratic_derivative(ak, labels[i]) * sygmoid_derivative(zk)
		for k := range exceptBiases {
			// Multiply an error by output of an appropriate hidden neuron
			// Correct a synapse immediately (Stochastic gradient)
			synapses[k][i] += cost * hiddenOut[i][k]
		}
		// Corrct biases
		// The gradient of the cost function with respect to the bias for each neuron is simply its error signal!
		synapses[HIDDEN][i] += cost
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
		log.Fatal(err)
	}
	labels, err := getTrainingLabels()
	if err != nil {
		log.Fatal(err)
	}
	// NOTE: forward and backward propagation implemented for a single data set item.
	// Also backward propagation supports only stochastic gradient and can't use batches of data items to learn
	fmt.Println(len(set), len(labels))
	prediction, hiddenOut := forward(set[0], synapses)
	backward(prediction, labels[0], hiddenOut, synapses)
}
