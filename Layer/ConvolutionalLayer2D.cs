using DeepDave.Helper;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System;
using System.Reflection;

namespace DeepDave.Layer
{
    public class ConvolutionalLayer2D : Layer2D
    {
        private int radius;

        public ConvolutionalLayer2D(int radius, int sliceCount, Layer2D prevLayer, string activationFunction) : base(prevLayer, activationFunction, sliceCount)
        {
            this.radius = radius;
            var x = (int)prevLayer.GetActivatedBuffer(0).Extent.X;
            var y = (int)prevLayer.GetActivatedBuffer(0).Extent.Y;

            this.variables = GPUHelper.accelerator.Allocate<float>(2);
            float[] vars = { Config.learningRate, radius};
            variables.CopyFrom(vars, 0, Index1.Zero, 2);

            for (int i = 0; i < sliceCount; i++) {
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.errors[i] = GPUHelper.CreateBuffer(x, y);
                this.weights[i] = GPUHelper.CreateBuffer(x, y, this.GetWeightCount());
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
            }
        }

        private int GetWeightCount() {
            return (radius * 2 + 1) * (radius * 2 + 1);
        }
    }
}
