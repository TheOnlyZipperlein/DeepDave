using DeepDave.Helper;
using ILGPU;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System;
using System.Drawing;
using System.Reflection;

namespace DeepDave.Layer {
    internal class InputLayer2D : Layer2D {
        private MemoryBuffer2D<float>[] inputs;

        internal InputLayer2D(Size dimensions, Layer2D prevLayer, int sliceCount, string function = null) : base(prevLayer, function, sliceCount) {
            this.variables = GPUHelper.accelerator.Allocate<float>(1);
            this.inputs = new MemoryBuffer2D<float>[sliceCount];
            var widthOutput = (int) dimensions.Width;
            var heightOutput = (int) dimensions.Height;
            for (int i = 0; i < sliceCount; i++) {
                this.inputs[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
                this.activated[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
                this.sumInput[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
            }
        }       

        internal MemoryBuffer2D<float>[] SwapInputs(MemoryBuffer2D<float>[] newInputs) {
            var buffer = inputs;
            inputs = newInputs;
            return buffer;
        }

        internal override void CalculateOutput() {
            if (activation == null) {
                activated = inputs;
            } else {
                var accelerator = GPUHelper.accelerator;
                for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                    sumCalculate(accelerator.DefaultStream, this.GetUnactivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), inputs[currentSlice], this.GetUnactivatedBuffer(currentSlice), this.bias[currentSlice], variables);
                }
                GPUHelper.Call.Wait();
            }
        }

        internal override void AdjustWeights() {
            //
        }

        internal override void CreateCalculateKernelInfo() {
            //
        }
        internal override void CalculateError(MemoryBuffer2D<float>[] shoulds) {
            //
        }
        internal override void CreateLearningKernelInfo() {
            //
        }
        internal override void CreateCalculateSigmaKernelInfo() {
            //
        }

    }
}
