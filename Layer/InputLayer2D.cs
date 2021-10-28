using DeepDave.Helper;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Drawing;
using System.Reflection;

namespace DeepDave.Layer {
    internal class InputLayer2D : Layer2D {
        private MemoryBuffer2D<float>[] inputs;
        private Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>> normalize;

        internal InputLayer2D(Size dimensions, Layer2D prevLayer, int sliceCount, string function = null) : base(prevLayer, null, sliceCount) {
            this.inputs = new MemoryBuffer2D<float>[sliceCount];
            var widthOutput = (int)dimensions.Width;
            var heightOutput = (int)dimensions.Height;
            for (int i = 0; i < sliceCount; i++) {
                this.inputs[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
                this.activated[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
                this.sumInput[i] = GPUHelper.CreateBuffer(widthOutput, heightOutput);
            }
            if (function != null) {
                MethodInfo methodInfo = Type.GetType("DeepDave.Layer.Kernels.NormalizationFunctions").GetMethod(function, BindingFlags.NonPublic | BindingFlags.Static);
                normalize = GPUHelper.CreateKernel(methodInfo).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>>();
            }
        }
        internal MemoryBuffer2D<float>[] SwapInputs(MemoryBuffer2D<float>[] newInputs) {
            var buffer = inputs;
            inputs = newInputs;
            return buffer;
        }
        internal override void CalculateOutput_() {
            if (normalize == null) {
                activated = inputs;
            } else {
                var accelerator = GPUHelper.accelerator;
                for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                    GPUHelper.Call.NormalizationFunction(normalize, inputs[currentSlice].Extent, inputs[currentSlice], activated[currentSlice], variable[currentSlice]);
                }
                GPUHelper.Call.Wait();
            }
        }

        internal override void ActivateOutput() {
            //
        }
        internal override void DerivateOutput() {
            //
        }
        internal override void AdjustWeights_() {
            //
        }
        internal override void CalculateError_(MemoryBuffer2D<float>[] shoulds) {
            //
        }
    }
}
