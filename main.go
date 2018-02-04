package main

import "log"

//import (
//"encoding/binary"
//"fmt"
//"io"
//"log"
//"math"
//"math/rand"
//"os"
//"time"
//)

//const TEST_SET_SIZE = 10000

//func getTrainigImages() (set [][]float64, err error) {
//fp, err := os.Open("t10k-images-idx3-ubyte")
//if err != nil {
//return
//}
//defer fp.Close()

//var magic int32
//err = binary.Read(fp, binary.BigEndian, &magic)
//if err != nil {
//return
//}
//if magic != 2051 {
//err = fmt.Errorf("Wrong magic number: %d. Expects: %d.", magic, 2051)
//return
//}

//var numImgs int32
//err = binary.Read(fp, binary.BigEndian, &numImgs)
//if err != nil {
//return
//}

//var rows int32
//err = binary.Read(fp, binary.BigEndian, &rows)
//if err != nil {
//return
//}

//var lines int32
//err = binary.Read(fp, binary.BigEndian, &lines)
//if err != nil {
//return
//}

//lim := int(numImgs)
//for i := 0; i < lim; i++ {
//image := make([]uint8, lines*rows)
//err = binary.Read(fp, binary.BigEndian, &image)
//if err != nil {
//break
//}
//var setItem []float64
//for _, pixel := range image {
//setItem = append(setItem, float64(pixel))
//}
//set = append(set, setItem)
//}

//if err == io.EOF {
//err = nil
//}
//return
//}

//func getTrainingLabels() (labels [][]float64, err error) {
//fp, err := os.Open("t10k-labels-idx1-ubyte")
//if err != nil {
//return
//}
//defer fp.Close()

//var magic int32
//err = binary.Read(fp, binary.BigEndian, &magic)
//if err != nil {
//return
//}

//if magic != 2049 {
//err = fmt.Errorf("Wrong magic number: %d. Expects: %d.", magic, 2049)
//return
//}

//var count int32
//err = binary.Read(fp, binary.BigEndian, &count)
//if err != nil {
//return
//}

//lim := int(count)
//for i := 0; i < lim; i++ {
//var labe uint8
//err = binary.Read(fp, binary.BigEndian, &labe)
//if err != nil {
//break
//}
//var hotEncoding []float64
//for i := 0; i < OUTPUT; i++ {
//if int(labe) == i {
//hotEncoding = append(hotEncoding, 1)
//} else {
//hotEncoding = append(hotEncoding, 0)
//}
//}
//labels = append(labels, hotEncoding)
//}
//if err == io.EOF {
//err = nil
//}
//return
//}

//const SCALING_BASE = 0.7
//const INPUT = 784
//const OUTPUT = 10
//const HIDDEN = 64
//const LEARNING_RATE = -.25
//const BIAS = .5

//// Create random 2D vector. Implement pre-init synapses
//func random() (out [][]float64) {
//rand.Seed(time.Now().UTC().UnixNano())
//for i := 0; i < HIDDEN; i++ {
//out = append(out, []float64{})
//for j := 0; j < OUTPUT; j++ {
//out[i] = append(out[i], rand.Float64()-0.5) // Float64 returns, as a float64, a pseudo-random number in [0.0,1.0).
//}
//}
//return
//}

//func NguyenWiderow() [][]float64 {
//beta := SCALING_BASE * math.Pow(HIDDEN, 1.0/INPUT)
//randSynapses := random()

//var norm float64
//for _, i := range randSynapses {
//norm = 0
//for _, j := range i {
//norm += j * j
//}
//norm = math.Sqrt(norm)
//for j, k := range i {
//i[j] = (k * beta) / norm
//}
//}
//return randSynapses
//}

//func addBiases(synapses [][]float64) [][]float64 {
//synapses = append(synapses, make([]float64, OUTPUT))
//for i := 0; i < OUTPUT; i++ {
//synapses[HIDDEN][i] = BIAS
//}
//return synapses
//}

//// For example Sygmoid
//func sygmoid(n float64) float64 {
//return 1 / (1 + math.Exp(n))
//}

//func forward(set []float64, synapses [][]float64) (output []float64, hiddenOut [][]float64) {
//var iSum, oSum float64

//// Each neuron of a first hidden layer receives all signals from input layer
//// and sums it. Input layer doesn'nt change input signal
//for _, i := range set {
//iSum += i * .00001 // Lowering of signal values to prevent overflow
//}

//iSum = sygmoid(iSum) // Activation of signal at a hidden layer
//lm := len(synapses)  // Count of neurons of a hidden layer apart from bias neuron

//for i := range synapses[0] {
//var outLine []float64
//oSum = 0

//for j := range synapses {
//jIOut := synapses[j][i] * iSum
//oSum += jIOut
//outLine = append(outLine, jIOut)
//}

//hiddenOut = append(hiddenOut, outLine)
//// Apply a bias
//oSum += synapses[lm-1][i] // Bias doesn't use weights. Bias is a weight without a signal.
//output = append(output, sygmoid(oSum))
//}

//return
//}

//func quadraticDerivative(a, e float64) float64 {
//return a - e
//}

//// n: weighted sum of j-1 layer activations. A.k.a. j layer k neuron input
//func sygmoidDerivative(n float64) float64 {
//return sygmoid(n) * (1 - sygmoid(n))
//}

//// TODO: add biases per layer
//// (ak - tk)g`(zk)aj
//// ak - output of an output layer k
//// zk - output layer k input (sum of hidden layer outputs)
//// tk - correct answer for an output neuron
//// aj - output of a hidden layer neuron
//func backward(out, labels []float64, hiddenOut, synapses [][]float64) [][]float64 {
//var cost, zk float64
//exceptBiases := synapses[:HIDDEN]
//for i, ak := range out { // outputs of an out layer
//zk = 0                            // out layer k neuron input (sum of a hidden layer outputs)
//for _, aj := range hiddenOut[i] { // Weighted outputs of a hidden layer k neuron
//// Count k neuron of out layer input (sum output layer input value)
//zk += aj
//}
//// Count an error derivative using delta rule
//cost = quadraticDerivative(ak, labels[i]) * sygmoidDerivative(zk)
//for k := range exceptBiases {
//// Multiply an error by output of an appropriate hidden neuron
//// Correct a synapse immediately (Stochastic gradient)
//// TODO: implement ability to learn in batches not ONLY stochastically
//synapses[k][i] += LEARNING_RATE * cost * hiddenOut[i][k]
//}
//// Correct biases
//// The gradient of the cost function with respect to the bias for each neuron is simply its error signal!
//synapses[HIDDEN][i] += cost
//}
//return synapses
//}

//func restore(id int) [][]float64 {
//return [][]float64{}
//}

//func quadraticCost(al, er []float64) float64 {
//var sum float64
//for i, out := range al {
//sum += math.Pow((out - er[i]), 2)
//}
//return sum * .5
//}

//func main() {
//synapses := NguyenWiderow()
//synapses = addBiases(synapses)
//set, err := getTrainigImages()
//if err != nil {
//log.Fatal(err)
//}
//labels, err := getTrainingLabels()
//if err != nil {
//log.Fatal(err)
//}
//// TODO: implement loop through the all data set and display cost for each iteration
//for i, v := range set {
//prediction, hiddenOut := forward(v, synapses)
//synapses = backward(prediction, labels[i], hiddenOut, synapses)
//fmt.Println(quadraticCost(prediction, labels[i]))
//}
//}
func main() {
	tLabels, err := getMNISTTrainingLabels("t10k-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}
	labels, err := getMNISTTrainingLabels("train-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}

	tSet, err := getMNISTTrainingImgs("t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	set, err := getMNISTTrainingImgs("train-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}

	nn := NewPerceptron(&sygmoid{}, &quadratic{})
	nn.Learn(set, labels)
	recognition := nn.Recognize(set)
	var l []float64
	for i, r := range recognition {
		l = tLabels[i]
		// Check for correctness:
		// Max prediction value index should appropriate 1 value index in labels one hot encoding
	}
}
