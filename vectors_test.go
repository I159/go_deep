package go_deep

import (
	"reflect"
	"testing"
)

func Test_dot1dTo2d(t *testing.T) {
	type args struct {
		d1 []float64
		d2 [][]float64
	}
	tests := []struct {
		name      string
		args      args
		wantD2out [][]float64
	}{
		{
			name: "transform",
			args: args{
				d1: []float64{
					1, 2, 3,
				},
				d2: [][]float64{
					{1, 2, 3, 4, 5},
					{10, 20, 30, 40, 50},
					{100, 200, 300, 400, 500},
				},
			},
			wantD2out: [][]float64{
				{1, 20, 300},
				{2, 40, 600},
				{3, 60, 900},
				{4, 80, 1200},
				{5, 100, 1500},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotD2out := dot1dTo2d(tt.args.d1, tt.args.d2); !reflect.DeepEqual(gotD2out, tt.wantD2out) {
				t.Errorf("dot1dTo2d() = %v, want %v", gotD2out, tt.wantD2out)
			}
		})
	}
}

func Test_appendAlongside(t *testing.T) {
	type args struct {
		d1 []float64
		d2 [][]float64
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			name: "appendAlongside",
			args: args{
				d1: []float64{1, 1, 1, 1, 1},
				d2: [][]float64{
					{1, 20, 300},
					{2, 40, 600},
					{3, 60, 900},
					{4, 80, 1200},
					{5, 100, 1500},
				},
			},
			want: [][]float64{
					{1, 20, 300, 1},
					{2, 40, 600, 1},
					{3, 60, 900, 1},
					{4, 80, 1200, 1},
					{5, 100, 1500, 1},
				},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := appendAlongside(tt.args.d1, tt.args.d2); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("appendAlongside() = %v, want %v", got, tt.want)
			}
		})
	}
}
