package go_deep

import (
	"reflect"
	"testing"
)

func Test_inputDense_forward(t *testing.T) {
	type fields struct {
		synapses      [][]float64
		nextLayerSize int
		currLayerSize int
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput [][]float64
	}{
		{
			name: "testForwardProp",
			fields: fields{
				synapses:      [][]float64{
					{1.0, 10.0, 100.0, 1000.0, 10000.0},
					{2.0, 20.0, 200.0, 2000.0, 20000.0},
					{3.0, 30.0, 300.0, 3000.0, 30000.0},
				},
				nextLayerSize: 5,
				currLayerSize: 3,
			},
			args:       args{[]float64{1, 2, 3}},
			wantOutput: [][]float64{
				{1.0, 4.0, 9.0},
				{10.0, 40.0, 90.0},
				{100.0, 400.0, 900.0},
				{1000.0, 4000.0, 9000.0},
				{10000.0, 40000.0, 90000.0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &inputDense{
				synapses:      tt.fields.synapses,
				nextLayerSize: tt.fields.nextLayerSize,
				currLayerSize: tt.fields.currLayerSize,
			}
			if gotOutput := l.forward(tt.args.input); !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("inputDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

func Test_hiddenDense_forward(t *testing.T) {
	type fields struct {
		activation         activation
		synapseInitializer synapseInitializer
		prevLayerSize      int
		currLayerSize      int
		nextLayerSize      int
		learningRate       float64
		corrections        [][]float64
		synapses           [][]float64
		activated          []float64
		input              []float64
	}
	type args struct {
		input [][]float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput [][]float64
		wantErr    bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &hiddenDense{
				activation:         tt.fields.activation,
				synapseInitializer: tt.fields.synapseInitializer,
				prevLayerSize:      tt.fields.prevLayerSize,
				currLayerSize:      tt.fields.currLayerSize,
				nextLayerSize:      tt.fields.nextLayerSize,
				learningRate:       tt.fields.learningRate,
				corrections:        tt.fields.corrections,
				synapses:           tt.fields.synapses,
				activated:          tt.fields.activated,
				input:              tt.fields.input,
			}
			gotOutput, err := l.forward(tt.args.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("hiddenDense.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("hiddenDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

func Test_outputDense_forward(t *testing.T) {
	type fields struct {
		activation    activation
		input         []float64
		cost          cost
		prevLayerSize int
	}
	type args struct {
		rowInput [][]float64
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantOutput []float64
		wantErr    bool
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &outputDense{
				activation:    tt.fields.activation,
				input:         tt.fields.input,
				cost:          tt.fields.cost,
				prevLayerSize: tt.fields.prevLayerSize,
			}
			gotOutput, err := l.forward(tt.args.rowInput)
			if (err != nil) != tt.wantErr {
				t.Errorf("outputDense.forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("outputDense.forward() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}

func Test_outputDense_forwardMeasure(t *testing.T) {
	type fields struct {
		activation    activation
		input         []float64
		cost          cost
		prevLayerSize int
	}
	type args struct {
		rowInput [][]float64
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
			l := &outputDense{
				activation:    tt.fields.activation,
				input:         tt.fields.input,
				cost:          tt.fields.cost,
				prevLayerSize: tt.fields.prevLayerSize,
			}
			gotPrediction, gotCost, err := l.forwardMeasure(tt.args.rowInput, tt.args.labels)
			if (err != nil) != tt.wantErr {
				t.Errorf("outputDense.forwardMeasure() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotPrediction, tt.wantPrediction) {
				t.Errorf("outputDense.forwardMeasure() gotPrediction = %v, want %v", gotPrediction, tt.wantPrediction)
			}
			if gotCost != tt.wantCost {
				t.Errorf("outputDense.forwardMeasure() gotCost = %v, want %v", gotCost, tt.wantCost)
			}
		})
	}
}
