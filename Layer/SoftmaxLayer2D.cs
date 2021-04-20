using DeepDave.Helper;
using ILGPU;
using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Text;

namespace DeepDave.Layer.Special {
    public class SoftmaxLayer2D : Layer2D {
        protected Action<AcceleratorStream, Index1, ArrayView<float>, ArrayView2D<float>> sumActivatedOutputs;
        protected Action<AcceleratorStream, Index2, ArrayView<float>, ArrayView2D<float>, ArrayView2D<float>> divisionBySumActivatedOutputs;
        protected Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>> softmaxDerivative;
        protected Action<AcceleratorStream, Index1, ArrayView2D<float>> softmaxNormalization;

        private MemoryBuffer2D<float>[] buffer;

        public SoftmaxLayer2D(Layer2D prevLayer, int sliceCount, Size outputSize) : base(prevLayer, null, sliceCount) {
            int x = outputSize.Width;
            int y = outputSize.Height;
            this.buffer = new MemoryBuffer2D<float>[sliceCount];

            for (int i = 0; i < sliceCount; i++) {
                this.bias[i] = GPUHelper.CreateBuffer(x, y);
                this.buffer[i] = GPUHelper.CreateBuffer(x, y);
                this.activated[i] = GPUHelper.CreateBuffer(x, y);
                this.sumInput[i] = GPUHelper.CreateBuffer(x, y);
                this.error[i] = GPUHelper.CreateBuffer(x, y);
                this.weight[i] = GPUHelper.CreateBuffer(x, y, GetWeightCount());
                this.variable[i] = GPUHelper.accelerator.Allocate<float>(2);
                float[] vars = { Config.learningRate, 0.0f};
                this.variable[i].CopyFrom(vars, Index1.Zero, Index1.Zero, this.variable[i].Extent);
                this.derived[i] = GPUHelper.CreateBuffer(x, y);
            }
        }

        protected override void InitializeKernels() {
            sumCalculateFunction = Type.GetType("DeepDave.Kernel.SumCalculate").GetMethod("FullyConnectedLayer2D", BindingFlags.NonPublic | BindingFlags.Static);
            activationFunction = Type.GetType("DeepDave.Kernel.ActivationFunctions").GetMethod("Softmax", BindingFlags.NonPublic | BindingFlags.Static);
            sumErrorFunction = Type.GetType("DeepDave.Kernel.SoftmaxFunctions").GetMethod("Error", BindingFlags.NonPublic | BindingFlags.Static);
            adjustWheigtsFunction = Type.GetType("DeepDave.Kernel.WheightAdjustment").GetMethod("FullyConnectedLayer2D", BindingFlags.NonPublic | BindingFlags.Static);

            var info = Type.GetType("DeepDave.Kernel.SoftmaxFunctions").GetMethod("SumActivatedOutputs", BindingFlags.NonPublic | BindingFlags.Static);
            sumActivatedOutputs = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index1, ArrayView<float>, ArrayView2D<float>>>();            
            info = Type.GetType("DeepDave.Kernel.SoftmaxFunctions").GetMethod("DivisionBySumActivatedOutputs", BindingFlags.NonPublic | BindingFlags.Static);
            divisionBySumActivatedOutputs = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView<float>, ArrayView2D<float>, ArrayView2D<float>>>();
            info = Type.GetType("DeepDave.Kernel.DerivativeFunctions").GetMethod("Softmax", BindingFlags.NonPublic | BindingFlags.Static);
            softmaxDerivative = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index2, ArrayView2D<float>, ArrayView2D<float>>>();
            info = Type.GetType("DeepDave.Kernel.SoftmaxFunctions").GetMethod("Normalization", BindingFlags.NonPublic | BindingFlags.Static);
            softmaxNormalization = GPUHelper.CreateKernel(info).CreateLauncherDelegate<Action<AcceleratorStream, Index1, ArrayView2D<float>>>();            
            base.InitializeKernels();
        }

        internal override void CalculateOutput_() {
            var accelerator = GPUHelper.accelerator;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                sumCalculate(accelerator.DefaultStream, this.GetUnactivatedBuffer(currentSlice).Extent, this.GetWeightBuffer(currentSlice), prevLayer.GetActivatedBuffer(currentSlice), this.GetUnactivatedBuffer(currentSlice), bias[currentSlice], variable[currentSlice]);
            }
            GPUHelper.Call.Wait();
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                softmaxNormalization(accelerator.DefaultStream, new Index1(1), this.GetUnactivatedBuffer(currentSlice));
            }
            GPUHelper.Call.Wait();
            this.ActivateOutput();
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                sumActivatedOutputs(accelerator.DefaultStream, new Index1(1), variable[currentSlice], buffer[currentSlice]);
            }
            GPUHelper.Call.Wait();
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                divisionBySumActivatedOutputs(accelerator.DefaultStream, this.activated[currentSlice].Extent, variable[currentSlice], buffer[currentSlice], activated[currentSlice]);
            }
            GPUHelper.Call.Wait();
        }
        internal override void ActivateOutput() {
            var accelerator = GPUHelper.accelerator;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                GPUHelper.Call.ActivationFunction(activation ,this.GetActivatedBuffer(currentSlice).Extent, this.GetUnactivatedBuffer(currentSlice), this.buffer[currentSlice], this.variable[currentSlice]);
            }            
            GPUHelper.Call.Wait();
        }

        internal void CalculateDerivative() { 
            var accelerator = GPUHelper.accelerator;
            for (int currentSlice = 0; currentSlice < activated.Length; currentSlice++) {
                softmaxDerivative(accelerator.DefaultStream, this.activated[currentSlice].Extent, this.GetActivatedBuffer(currentSlice), this.GetDerived(currentSlice));
            }
            GPUHelper.Call.Wait();
        }

        internal override void CalculateError_(MemoryBuffer2D<float>[] shoulds) {
            CalculateDerivative();
            base.CalculateError(shoulds);
        }
        private int GetWeightCount() {
            var x = Convert.ToInt32(prevLayer.GetActivatedBuffer(0).Width * prevLayer.GetActivatedBuffer(0).Height);
            return x;
        }
        #region unarm Layer2D construction
        internal override void CreateActivationFunctionInfo() {
            //
        }
        internal override void CreateCalculateKernelInfo() {
            //
        }
        internal override void CreateCalculateErrorKernelInfo() {
            //
        }

        internal override void CreateAdjustWeightsKernelInfo() {
            //
        }
        #endregion
    }
}