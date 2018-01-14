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
