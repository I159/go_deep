package goDeep

import "fmt"

func areCorrsConsistent(corrSize, layerSize, synapsesSize int) error {
	if corrSize != layerSize || layerSize != corrSize {
		return locatedError{
			fmt.Sprintf(
				"Synapses, corrections and a current layer size are not consistent.\nCorrections: %d\nSynapses:%d\nLayer: %d\n",
				corrSize, layerSize, synapsesSize,
			),
		}
	}
	return nil
}
func checkSynapsesSize(layerSize, synapsesSize int) error {
	if layerSize == 0 || synapsesSize == 0 || synapsesSize != layerSize {
		return locatedError{
			fmt.Sprintf(
				"Synapses vector is not appropriate size to a current layer size.\nLayer size: %d\nSynapses size: %d",
				synapsesSize,
				layerSize,
			),
		}
	}
	return nil
}

func checkInputSize(inputSize, layerSize int) error {
	if inputSize != layerSize {
		return locatedError{
			fmt.Sprintf(
				"Input is not appropriate size to a current layer size.\nLayer size: %d\nInput size: %d",
				layerSize,
				inputSize,
			),
		}
	}
	return nil
}

func areSizesConsistent(inputSize, layerSize, synapsesSize int, bias bool) (err error) {
	if err = checkSynapsesSize(layerSize, synapsesSize); err == nil {
		if bias {
			layerSize--
		}
		err = checkInputSize(inputSize, layerSize)
	}
	return
}
