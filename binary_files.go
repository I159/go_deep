package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

func getMNISTTrainingImgs(path string) (set [][]float64, err error) {
	fp, err := os.Open(path)
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

	var numImgs int32
	err = binary.Read(fp, binary.BigEndian, &numImgs)
	if err != nil {
		return
	}

	var rows int32
	err = binary.Read(fp, binary.BigEndian, &rows)
	if err != nil {
		return
	}

	var lines int32
	err = binary.Read(fp, binary.BigEndian, &lines)
	if err != nil {
		return
	}

	lim := int(numImgs)
	for i := 0; i < lim; i++ {
		image := make([]uint8, lines*rows)
		err = binary.Read(fp, binary.BigEndian, &image)
		if err != nil {
			break
		}
		var setItem []float64
		for _, pixel := range image {
			setItem = append(setItem, float64(pixel))
		}
		set = append(set, setItem)
	}

	if err == io.EOF {
		err = nil
	}
	return
}

func getMNISTTrainingLabels(path string, labeCount int) (labels [][]float64, err error) {
	// Expect coherent integer labels starting from 0. Labels count is overall count of
	// coherent labels list including 0.
	fp, err := os.Open(path)
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

	lim := int(count)
	for i := 0; i < lim; i++ {
		var labe uint8
		err = binary.Read(fp, binary.BigEndian, &labe)
		if err != nil {
			break
		}
		var hotEncoding []float64
		for i := 0; i < labeCount; i++ {
			if int(labe) == i {
				hotEncoding = append(hotEncoding, 1)
			} else {
				hotEncoding = append(hotEncoding, 0)
			}
		}
		labels = append(labels, hotEncoding)
	}
	if err == io.EOF {
		err = nil
	}
	return
}
