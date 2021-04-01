using DeepDave.Helper;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Text;

namespace DeepDave.Layer {
    public class FullyConnectedLayer2D : Layer2D {        
        public FullyConnectedLayer2D(Size outputSize, int sliceCount, Layer2D prevLayer, string activationFunction) : base(prevLayer, activationFunction, sliceCount) {           

            var x = (int)outputSize.Width;
            var y = (int)outputSize.Height;

            this.variables = GPUHelper.accelerator.Allocate<float>(2);
            variables.CopyFrom(Config.learningRate, Index1.Zero);

            for(int i=0; i<sliceCount; i++) {
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.errors[i] = GPUHelper.CreateBuffer(x, y);
                this.weights[i] = GPUHelper.CreateBuffer(x, y, this.GetWeightCount());
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
            }  
        }

        private int GetWeightCount() { 
            var x = Convert.ToInt32(prevLayer.GetActivatedBuffer(0).Width * prevLayer.GetActivatedBuffer(0).Height);
            return x;
        }
    }
}
