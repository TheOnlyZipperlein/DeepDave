using DeepDave.Helper;
using System;
using System.Drawing;

namespace DeepDave.Layer {
    public class FullyConnectedLayer2D : Layer2D {
        public FullyConnectedLayer2D(Size outputSize, int sliceCount, Layer2D prevLayer, string activationFunction) : base(prevLayer, activationFunction, sliceCount) {
            var x = (int)outputSize.Width;
            var y = (int)outputSize.Height;
            var fac = GetSuitableFactorForFunction(function, x * y);
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, fac };
                this.variable[i] = GPUHelper.CreateBuffer(source, 2);
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, this.GetWeightCount());
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
            }
        }

        private int GetWeightCount() {
            var x = Convert.ToInt32(prevLayer.GetActivatedBuffer(0).Width * prevLayer.GetActivatedBuffer(0).Height);
            return x;
        }
    }
}
