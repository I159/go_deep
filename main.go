package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

func main() {
	fmt.Println("OOooOOOOOooooo")
	tLabels, err := getMNISTTrainingLabels("t10k-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}
	labels, err := getMNISTTrainingLabels("train-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}

	tSet, err := getMNISTTrainingImgs("t10k-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	set, err := getMNISTTrainingImgs("train-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}

	nn := NewPerceptron(.25, &sygmoid{}, &quadratic{}, 784, 64, 10)

	learnCost := nn.Learn(set, labels)

	f, err := os.Create("learn_costs.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	fmt.Fprint(w, learnCost)

	// FIXME: this is not a cost! This is a hidden layer output!
	recognition, recCost := nn.Recognize(tSet)

	f, err = os.Create("recognition_costs.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w = bufio.NewWriter(f)
	fmt.Fprint(w, recCost)

	for i, rec := range recognition {
		for j, corr := range tLabels[i] {
			fmt.Println(rec[j], corr)
		}
	}
}
