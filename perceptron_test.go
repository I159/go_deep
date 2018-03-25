package goDeep

import (
	"reflect"
	"testing"
)

func TestPerceptron_backward(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		prediction []float64
		labels     []float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			if err := n.backward(tt.args.prediction, tt.args.labels); (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.backward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPerceptron_applyCorrections(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		batchSize float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			if err := l.applyCorrections(tt.args.batchSize); (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.applyCorrections() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPerceptron_Learn(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		set       [][]float64
		labels    [][]float64
		epochs    int
		batchSize int
	}
	tests := []struct {
		name             string
		fields           fields
		args             args
		wantCostGradient []float64
		wantErr          bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			gotCostGradient, err := n.Learn(tt.args.set, tt.args.labels, tt.args.epochs, tt.args.batchSize)
			if (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.Learn() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotCostGradient, tt.wantCostGradient) {
				t.Errorf("Perceptron.Learn() = %v, want %v", gotCostGradient, tt.wantCostGradient)
			}
		})
	}
}

func TestPerceptron_forward(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		rowInput []float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []float64
		wantErr bool
	}{
		{
			name: "network_forward",
			fields: fields{
				input: &inputDense{},
				hidden: []hiddenLayer{
					&hiddenDense{},
				},
				output: &outputDense{},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			got, err := n.forward(tt.args.rowInput)
			if (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Perceptron.forward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPerceptron_forwardMeasure(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		rowInput []float64
		labels   []float64
	}
	tests := []struct {
		name           string
		fields         fields
		args           args
		wantPrediction []float64
		wantCost       float64
		wantErr        bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			gotPrediction, gotCost, err := n.forwardMeasure(tt.args.rowInput, tt.args.labels)
			if (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.forwardMeasure() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPrediction, tt.wantPrediction) {
				t.Errorf("Perceptron.forwardMeasure() gotPrediction = %v, want %v", gotPrediction, tt.wantPrediction)
			}
			if gotCost != tt.wantCost {
				t.Errorf("Perceptron.forwardMeasure() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}

func TestPerceptron_Recognize(t *testing.T) {
	type fields struct {
		input  inputLayer
		hidden []hiddenLayer
		output outputLayer
	}
	type args struct {
		set [][]float64
	}
	tests := []struct {
		name           string
		fields         fields
		args           args
		wantPrediction [][]float64
		wantErr        bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Perceptron{
				input:  tt.fields.input,
				hidden: tt.fields.hidden,
				output: tt.fields.output,
			}
			gotPrediction, err := n.Recognize(tt.args.set)
			if (err != nil) != tt.wantErr {
				t.Errorf("Perceptron.Recognize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPrediction, tt.wantPrediction) {
				t.Errorf("Perceptron.Recognize() = %v, want %v", gotPrediction, tt.wantPrediction)
			}
		})
	}
}
