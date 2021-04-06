using DeepDave.Helper;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System;
using System.Reflection;

namespace DeepDave.Layer { 
    public class ConvolutionalLayer2D : Layer2D {
        private int radius;
        private float fac;
        public ConvolutionalLayer2D(int radius, int sliceCount, Layer2D prevLayer, string activationFunction) : base(prevLayer, activationFunction, sliceCount) {
            this.radius = radius;             
            var x = (int) prevLayer.GetActivatedBuffer(0).Extent.X;
            var y = (int) prevLayer.GetActivatedBuffer(0).Extent.Y;
            this.fac = GetSuitableFactorForFunction(function, (2*radius + 1)*(2* radius + 1));
            for (int i = 0; i < sliceCount; i++) {
                float[] source = { Config.learningRate, fac , radius };
                this.variable[i] = GPUHelper.CreateBuffer(source, source.Length);
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, this.GetWeightCount());
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
            }
        }
        private int GetWeightCount() {
            return (radius * 2 + 1) * (radius * 2 + 1);
        }
    }
}
